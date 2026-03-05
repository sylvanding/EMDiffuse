import argparse
import os
import socket
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


def main(opt):
    local_rank = opt['local_rank']
    is_main = (opt['global_rank'] == 0)

    Util.set_seed(opt['seed'] + local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    if not is_main:
        phase_writer.writer = None

    if is_main:
        phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    phase_loader, val_loader = define_dataloader(phase_logger, opt)
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    if is_main:
        phase_logger.info('Begin model {}.'.format(opt['phase']))

    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _ddp_worker(local_rank, world_size, port, opt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    opt['distributed'] = True
    opt['global_rank'] = local_rank
    opt['local_rank'] = local_rank
    opt['world_size'] = world_size
    try:
        main(opt)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/EMDiffuse-n.json',
                        help='JSON file for configuration')
    parser.add_argument('--path', type=str, default=None, help='patch of cropped patches')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('--gpu', type=str, default=None, help='the gpu devices used')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-z', '--z_times', default=None, type=int, help='The anisotropy time of the volume em')
    parser.add_argument('--mean', type=int, default=2,
                        help='EMDiffuse samples one plausible solution from distribution. The number of samples you '
                             'want to generate and averaging')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config if set)')
    parser.add_argument('--step', type=int, default=None, help='Steps of the diffusion process. More steps lead to '
                                                               'better image quality. ')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume state path and load epoch number e.g., experiments/EMDiffuse-n/2720')
    parser.add_argument('--tile', type=int, default=None,
                        help='Enable tiled inference on full images with given tile size (e.g., 256)')
    parser.add_argument('--tile_overlap', type=int, default=64,
                        help='Overlap pixels between adjacent tiles (default: 64)')

    args = parser.parse_args()
    opt = Praser.parse(args)

    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    n_gpu = len(opt['gpu_ids'])

    if n_gpu > 1 and opt['phase'] == 'train':
        port = _find_free_port()
        print(f'Launching DDP training on {n_gpu} GPUs (port {port})')
        mp.spawn(_ddp_worker, args=(n_gpu, port, opt), nprocs=n_gpu, join=True)
    else:
        main(opt)
