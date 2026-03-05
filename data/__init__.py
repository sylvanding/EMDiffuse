from functools import partial
import numpy as np

from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj



def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader """
    dataloader_args = dict(opt['datasets'][opt['phase']]['dataloader']['args'])
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    if opt.get('tile') is not None:
        dataloader_args['batch_size'] = 1
        dataloader_args['shuffle'] = False
        dataloader_args.pop('drop_last', None)
        logger.info('Tile mode: forcing batch_size=1')

    phase_dataset, val_dataset = define_dataset(logger, opt)

    if dataloader_args.get('num_workers', 0) > 0:
        dataloader_args['persistent_workers'] = True

    train_sampler = None
    if opt.get('distributed') and opt['phase'] == 'train':
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            phase_dataset,
            num_replicas=opt['world_size'],
            rank=opt['global_rank'],
            shuffle=dataloader_args.pop('shuffle', True),
        )
        dataloader_args.pop('shuffle', None)
    dataloader = DataLoader(phase_dataset, sampler=train_sampler,
                            worker_init_fn=worker_init_fn, **dataloader_args)

    if val_dataset is not None:
        val_args = dict(opt['datasets'][opt['phase']]['dataloader'].get('val_args', {}))
        val_args.setdefault('pin_memory', True)
        val_args['num_workers'] = 0
        val_args.pop('persistent_workers', None)
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **val_args)
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    if opt['phase'] != 'train':
        if opt['task'] == '3d_reconstruction':
            from vEM_test_pre import recon_pre
            dataset_opt['args']['data_root'] = recon_pre(dataset_opt['args']['data_root'])

    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = None

    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    val_max_per_cell = dataloder_opt.get('val_max_per_cell', 0)

    if val_max_per_cell > 0 and opt['phase'] == 'train':
        import copy
        val_opt = copy.deepcopy(dataset_opt)
        val_opt['args']['max_per_cell'] = val_max_per_cell
        val_opt['args']['phase'] = 'val'
        val_dataset = init_obj(val_opt, logger, default_file_name='data.dataset', init_type='Val Dataset')
        valid_len = len(val_dataset)

        if 'debug' in opt['name'] and isinstance(data_len, int) and data_len < len(phase_dataset):
            phase_dataset, _ = subset_split(dataset=phase_dataset, lengths=[data_len, len(phase_dataset) - data_len],
                                            generator=Generator().manual_seed(opt['seed']))
    else:
        valid_split = dataloder_opt.get('validation_split', 0)
        if valid_split > 0.0 or 'debug' in opt['name']:
            if isinstance(valid_split, int):
                assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
                valid_len = valid_split
            else:
                valid_len = int(data_len * valid_split)
            data_len -= valid_len
            phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len],
                                                      generator=Generator().manual_seed(opt['seed']))

    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))
    return phase_dataset, val_dataset


def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length: offset]))
    return Subsets
