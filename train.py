import os
import sys
import math

import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

import random
import numpy as np
import torch
import torch.distributed as distributed

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

from hyram_vps.model.trainer import Trainer
from hyram_vps.dataset.setup_training_data import setup_main_training_datasets
from hyram_vps.utils.logger import TensorboardLogger

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


def info_if_rank_zero(msg):
    if local_rank == 0:
        log.info(msg)


@hydra.main(version_base='1.3.2', config_path='hyram_vps/config', config_name='train_sunseg_config.yaml')
def train(cfg: DictConfig):
    # initial setup
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir
    info_if_rank_zero(f'All configuration: {cfg}')
    info_if_rank_zero(f'Number of detected GPUs: {num_gpus}')

    # cuda setup
    torch.cuda.set_device(local_rank)
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # number of dataloader workers
    cfg.num_workers //= num_gpus
    info_if_rank_zero(f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(run_dir, logging.getLogger(), enabled_tb=(local_rank == 0))

    # SUN-SEG uses a single video-training stage.
    stages = ['main_training']
    for stage in stages:
        # Set seeds to ensure the same initialization
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # setting up configurations
        stage_cfg = cfg[stage]
        info_if_rank_zero(f'Training stage: {stage}')
        info_if_rank_zero(f'Training configuration: {stage_cfg}')
        stage_cfg.batch_size //= num_gpus
        info_if_rank_zero(f'Batch size (per GPU): {stage_cfg.batch_size}')

        # construct the trainer
        trainer = Trainer(cfg, stage_cfg, log=log, run_path=run_dir).train()

        # load previous checkpoint if needed
        if cfg['checkpoint'] is not None:
            curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
            cfg['checkpoint'] = None
            info_if_rank_zero('Model checkpoint loaded!')
        else:
            curr_iter = 0

        # load previous network weights if needed
        if cfg['weights'] is not None:
            info_if_rank_zero('Loading weights from the disk')
            trainer.load_weights(cfg['weights'])
            cfg['weights'] = None

        # determine time to change max skip
        total_iterations = stage_cfg['num_iterations']
        if 'max_skip_schedule' in stage_cfg:
            max_skip_schedule = stage_cfg['max_skip_schedule']
            increase_skip_fraction = stage_cfg['max_skip_schedule_fraction']
            change_skip_iter = [round(total_iterations * f) for f in increase_skip_fraction]
            # Skip will only change after an epoch, not in the middle
            log.info(f'The skip value will change at these iters: {change_skip_iter}')
        else:
            change_skip_iter = []
        change_skip_iter.append(total_iterations + 1)  # dummy value to avoid out of index error

        # setup datasets
        dataset, sampler, loader = setup_main_training_datasets(cfg, max_skip_schedule[0])
        max_skip_schedule = max_skip_schedule[1:]
        change_skip_iter = change_skip_iter[1:]
        log.info(f'Number of training samples: {len(dataset)}')
        log.info(f'Number of training batches: {len(loader)}')

        # determine max epoch
        total_epoch = math.ceil(total_iterations / len(loader))
        current_epoch = curr_iter // len(loader)
        log.info(f'We will approximately use {total_epoch} epochs.')

        # training loop
        try:
            # Need this to select random bases in different workers
            np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
            while curr_iter < total_iterations:
                # Crucial for randomness!
                sampler.set_epoch(current_epoch)
                current_epoch += 1
                log.debug(f'Current epoch: {current_epoch}')

                trainer.train()
                for data in loader:
                    # Update skip if needed
                    if curr_iter >= change_skip_iter[0]:
                        while curr_iter >= change_skip_iter[0]:
                            cur_skip = max_skip_schedule[0]
                            max_skip_schedule = max_skip_schedule[1:]
                            change_skip_iter = change_skip_iter[1:]
                        log.info(f'Changing max skip to {cur_skip=}')
                        _, sampler, loader = setup_main_training_datasets(cfg, cur_skip)
                        break

                    trainer.do_pass(data, curr_iter)
                    curr_iter += 1

                    if curr_iter >= total_iterations:
                        break
        finally:
            if not cfg.debug:
                trainer.save_weights(curr_iter)
                trainer.save_checkpoint(curr_iter)

        torch.cuda.empty_cache()
    # clean-up
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
