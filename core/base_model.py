import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn
import torch.distributed as dist


import core.util as Util
CustomResult = collections.namedtuple('CustomResult', 'name result')

class BaseModel():

    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer):
        self.opt = opt
        self.phase = opt['phase']
        self.set_device = Util.set_device
        self.mean = opt['mean'] if 'mean' in opt.keys() else 1
        self.schedulers = []
        self.optimizers = []
        self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = 0
        self.transfer_epoch = 0
        self.iter = 0
        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([],[])

        self.is_distributed = opt.get('distributed', False)
        self.is_main = (opt.get('global_rank', 0) == 0)

    def train(self):
        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1

            if self.is_distributed and hasattr(self.phase_loader, 'sampler'):
                sampler = self.phase_loader.sampler
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(self.epoch)

            train_log = self.train_step()

            if self.is_main:
                print('epoch {} / {} | iter {} / {}'.format(
                    self.epoch, int(self.opt['train']['n_epoch']),
                    self.iter, int(self.opt['train']['n_iter'])))
                train_log.update({'epoch': self.epoch, 'iters': self.iter})

                for key, value in train_log.items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))

            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                if self.is_main:
                    self.logger.info('Saving checkpoint at epoch {}'.format(self.epoch))
                    self.save_everything()
                if self.is_distributed:
                    dist.barrier()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                if self.is_main:
                    if self.val_loader is None:
                        self.logger.warning('Validation skipped: val_loader is None.')
                    else:
                        self.logger.info('--- Validation Start (epoch {}) ---'.format(self.epoch))
                        val_log = self.val_step()
                        if val_log:
                            for key, value in val_log.items():
                                self.logger.info('{:5s}: {}\t'.format(str(key), value))
                            self.logger.info('--- Validation End ---')
                if self.is_distributed:
                    dist.barrier()

        if self.is_main:
            self.logger.info('Training finished. epoch={}, iter={}'.format(self.epoch, self.iter))

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        net = network.module if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else network
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state'] is None:
            return 
        self.logger.info('Begin loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self.opt['path']['resume_state'], network_label)
        
        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        net = network.module if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else network
        net.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: Util.set_device(storage)), strict=strict)

    def save_training_state(self):
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self.opt['path']['resume_state'] is None:
            return
        self.logger.info('Begin loading training states')
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self.opt['path']['resume_state'])
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        if len(resume_schedulers)== len(self.schedulers):
            for i, s in enumerate(resume_schedulers):
                self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.transfer_epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')
