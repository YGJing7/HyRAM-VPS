import os
from os import path
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from hyram_vps.dataset.vos_dataset import VOSMergeTrainDataset

_local_rank = int(os.environ['LOCAL_RANK'])
log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + _local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {_local_rank}')


def setup_main_training_datasets(cfg, max_skip):
    root = cfg.data.vos_datasets.base
    sunseg_cfg = cfg.data.vos_datasets.SUNSEG_TRAIN
    dataset_configs = {
        'SUNSEG_TRAIN': {
            'im_root': path.join(root, sunseg_cfg.image_directory),
            'gt_root': path.join(root, sunseg_cfg.mask_directory),
            'max_skip': max_skip // sunseg_cfg.frame_interval,
            'subset': None,
            'empty_masks': None,
            'multiplier': sunseg_cfg.multiplier,
        }
    }

    dataset = VOSMergeTrainDataset(dataset_configs,
                                   seq_length=cfg.main_training.seq_length,
                                   max_num_obj=cfg.main_training.num_objects,
                                   size=cfg.main_training.crop_size[0], 
                                   merge_probability=cfg.main_training.merge_probability)

    batch_size = cfg.main_training.batch_size
    num_workers = cfg.num_workers
    sampler, loader = construct_loader(dataset, batch_size, num_workers, _local_rank)

    log.info(f'Using a max skip of {max_skip} frames')

    return dataset, sampler, loader


def construct_loader(dataset, batch_size, num_workers, local_rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    rank=local_rank,
                                                                    shuffle=True)
    train_loader = DataLoader(dataset,
                              batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              drop_last=True,
                              persistent_workers=True)
    return train_sampler, train_loader
