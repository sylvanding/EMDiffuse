import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base_model import BaseModel
from core.logger import LogTracker
import core.util as Util


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DiReP(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(DiReP, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        self.netG = self.set_device(self.netG)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA)
        self.load_networks()
        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training()

        self.netG.set_loss(self.loss_fn)
        self.netG.set_new_noise_schedule(phase=self.phase)
        if self.ema_scheduler is not None:
            self.netG_EMA.set_new_noise_schedule(phase=self.phase)

        if self.is_distributed and self.phase == 'train':
            self.netG = nn.parallel.DistributedDataParallel(
                self.netG, device_ids=[self.opt['local_rank']])
            self.logger.info('Wrapped netG with DDP on rank {}'.format(self.opt['local_rank']))

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

    @property
    def _netG_module(self):
        """Unwrap DataParallel/DDP to access the raw Network module."""
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.netG.module
        return self.netG

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.img_min = data.get('img_min', None)
        self.img_max = data.get('img_max', None)
        self.gt_min = data.get('gt_min', None)
        self.gt_max = data.get('gt_max', None)
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu() + 1) / 2,
            'cond_image': (self.cond_image.detach()[:].float().cpu() + 1) / 2,
        }
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu() + 1) / 2
            })

        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.output[idx].detach().float().cpu())
            ret_path.append('Input_{}'.format(self.path[idx]))
            ret_result.append(self.cond_image[idx].detach().float().cpu())
            if self.mean > 1 and hasattr(self, 'outputs') and self.outputs:
                for output_index in range(len(self.outputs)):
                    ret_path.append('Out_round{}_{}'.format(output_index, self.path[idx]))
                    ret_result.append(self.outputs[output_index][idx].detach().float().cpu())
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        if self.is_main:
            from tqdm import tqdm
            pbar = tqdm(self.phase_loader, desc=f'Epoch {self.epoch}',
                        leave=True, dynamic_ncols=True)
        else:
            pbar = self.phase_loader
        for train_data in pbar:
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.is_main and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f'{loss.item():.4f}', iter=self.iter)
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self._netG_module)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        ddim_steps = self.opt['train'].get('val_sample_steps', None)
        max_val_images = self.opt['train'].get('max_val_images', 4)

        with torch.no_grad():
            val_data = next(iter(self.val_loader))
            self.set_input(val_data)

            self.output, self.visuals = self._netG_module.restoration(
                self.cond_image, sample_num=self.sample_num,
                y_0=self.gt_image, ddim_steps=ddim_steps)

            self.writer.set_iter(self.epoch, self.epoch, phase='val')

            for met in self.metrics:
                key = met.__name__
                value = met(self.gt_image, self.output)
                self.val_metrics.update(key, value)
                self.writer.add_scalar(key, value)

            n = min(max_val_images, self.batch_size)
            for key, value in self.get_current_visuals(phase='val').items():
                self.writer.add_images(key, value[:n])

        torch.cuda.empty_cache()
        return self.val_metrics.result()

    def _get_net_for_inference(self):
        """Return the best available network for inference (prefer EMA)."""
        if self.ema_scheduler is not None and hasattr(self, 'netG_EMA'):
            return self.netG_EMA
        return self._netG_module

    def model_test(self, sample_num):
        net = self._get_net_for_inference()
        output, self.visuals = net.restoration(self.cond_image,
                                               sample_num=sample_num)
        return output

    @staticmethod
    def _make_blend_weight(tile_h, tile_w, overlap, device,
                           has_top=True, has_bottom=True, has_left=True, has_right=True):
        """Create a 2D blending weight with linear ramps only on sides that overlap neighbors."""
        weight = torch.ones(tile_h, tile_w, device=device)
        if overlap <= 0:
            return weight
        ramp_up = torch.linspace(0, 1, overlap + 1, device=device)[1:]   # (0, 1] excluding 0
        ramp_down = ramp_up.flip(0)
        if has_top:
            weight[:overlap, :] *= ramp_up.unsqueeze(1)
        if has_bottom:
            weight[-overlap:, :] *= ramp_down.unsqueeze(1)
        if has_left:
            weight[:, :overlap] *= ramp_up.unsqueeze(0)
        if has_right:
            weight[:, -overlap:] *= ramp_down.unsqueeze(0)
        return weight

    def _tiled_inference(self, cond_image, sample_num, tile_size, overlap):
        """Run diffusion on a single full-size image using overlapping tiles.

        Args:
            cond_image: (1, C, H, W) tensor in [-1, 1]
            sample_num: number of intermediate samples to record
            tile_size: spatial size of each tile
            overlap: overlap pixels between adjacent tiles

        Returns:
            output: (1, C, H, W) blended result
        """
        net = self._get_net_for_inference()
        _, C, H, W = cond_image.shape
        stride = tile_size - overlap

        pad_h = (stride - (H - tile_size) % stride) % stride if H > tile_size else tile_size - H
        pad_w = (stride - (W - tile_size) % stride) % stride if W > tile_size else tile_size - W
        cond_padded = F.pad(cond_image, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, pH, pW = cond_padded.shape

        output_acc = torch.zeros(1, C, pH, pW, device=cond_image.device)
        weight_acc = torch.zeros(1, 1, pH, pW, device=cond_image.device)

        tiles_y = list(range(0, pH - tile_size + 1, stride))
        tiles_x = list(range(0, pW - tile_size + 1, stride))

        total_tiles = len(tiles_y) * len(tiles_x)
        self.logger.info(f'Tiled inference: image {H}x{W}, padded {pH}x{pW}, '
                         f'{len(tiles_y)}x{len(tiles_x)}={total_tiles} tiles '
                         f'(tile={tile_size}, overlap={overlap})')

        tile_idx = 0
        for iy, y0 in enumerate(tiles_y):
            for ix, x0 in enumerate(tiles_x):
                tile_idx += 1
                blend_w = self._make_blend_weight(
                    tile_size, tile_size, overlap, cond_image.device,
                    has_top=(iy > 0),
                    has_bottom=(iy < len(tiles_y) - 1),
                    has_left=(ix > 0),
                    has_right=(ix < len(tiles_x) - 1),
                )
                tile_cond = cond_padded[:, :, y0:y0+tile_size, x0:x0+tile_size]
                tile_out, _ = net.restoration(tile_cond, sample_num=sample_num)
                output_acc[:, :, y0:y0+tile_size, x0:x0+tile_size] += tile_out * blend_w
                weight_acc[:, :, y0:y0+tile_size, x0:x0+tile_size] += blend_w
                if tile_idx % 10 == 0 or tile_idx == total_tiles:
                    print(f'  tile {tile_idx}/{total_tiles}')

        output_acc /= weight_acc.clamp(min=1e-8)
        return output_acc[:, :, :H, :W]

    def test(self):
        self.netG.eval()
        if self.ema_scheduler is not None and hasattr(self, 'netG_EMA'):
            self.netG_EMA.eval()
        self.test_metrics.reset()

        tile_size = self.opt.get('tile')
        tile_overlap = self.opt.get('tile_overlap', 64)
        use_tiling = tile_size is not None

        with torch.no_grad():
            for phase_data in self.phase_loader:
                self.set_input(phase_data)

                if use_tiling:
                    self.outputs = []
                    batch_means = []
                    batch_stds = []
                    for b_idx in range(self.batch_size):
                        single_cond = self.cond_image[b_idx:b_idx+1]
                        sample_outputs = []
                        for i in range(self.mean):
                            out = self._tiled_inference(single_cond, self.sample_num,
                                                        tile_size, tile_overlap)
                            sample_outputs.append(out)
                            self.outputs.append(out)
                        stacked = torch.cat(sample_outputs, dim=0)
                        batch_means.append(stacked.mean(dim=0, keepdim=True))
                        batch_stds.append(stacked.std(dim=0, keepdim=True)
                                          if self.mean > 1
                                          else torch.zeros_like(sample_outputs[0]))
                    self.output = torch.cat(batch_means, dim=0)
                    self.model_uncertainty = torch.cat(batch_stds, dim=0)
                else:
                    self.outputs = []
                    for i in range(self.mean):
                        output = self.model_test(self.sample_num)
                        self.outputs.append(output)
                    if self.mean > 1:
                        self.output = torch.stack(self.outputs, dim=0).mean(dim=0)
                        self.model_uncertainty = torch.stack(self.outputs, dim=0).std(dim=0)
                    else:
                        self.output = self.outputs[0]
                        self.model_uncertainty = torch.zeros_like(self.output)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(), norm=self.opt['norm'])

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label + '_ema', strict=False)

    def save_everything(self):
        netG_label = self._netG_module.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label + '_ema')
        self.save_training_state()
