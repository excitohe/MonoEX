import logging

import numpy as np
import torch.utils.data
from monoex.utils.comm import get_world_size
from monoex.utils.env import import_file, seed_all_rng

from . import datasets as D
from . import samplers
from .collate import BatchCollator
from .transforms import build_transforms


def build_dataset(cfg, transforms, dataset_catalog, is_train=True):
    """
    Args:
        dataset_list (list[str]): Contains the names of the datasets.
        transforms (callable): Transforms to apply to each (image, target)
            sample construct a dataset.
        dataset_catalog (DatasetCatalog): Contains the infomation on 
            how to construct a dataset.
        is_train (bool): Whether to setup the dataset for training or testing.
    Returns:
    """
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, " "got {}".format(dataset_list))
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg.DATASETS.STYLE)
        factory = getattr(D, data["factory"])
        args = data["args"]

        args["cfg"] = cfg
        args["is_train"] = is_train
        args["transforms"] = transforms
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def build_data_loader(cfg, is_train=True):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, (
            f"SOLVER.IMS_PER_BATCH ({images_per_batch}) must be "
            f"divisible by the number of GPUs ({num_gpus}) used."
        )
        images_per_gpu = images_per_batch // num_gpus
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, (
            f"SOLVER.IMS_PER_BATCH ({images_per_batch}) must be "
            f"divisible by the number of GPUs ({num_gpus}) used."
        )
        images_per_gpu = images_per_batch // num_gpus

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    path_catalog = import_file("monoex.config.catalogs", cfg.PATHS_CATALOG, True)
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(cfg.DATALOADER.SAMPLER))
        if cfg.DATALOADER.SAMPLER == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(dataset))
        else:
            raise ValueError(f"Unsupport sampler {cfg.DATALOADER.SAMPLER}")
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_gpu, drop_last=True)
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
        data_loaders.append(data_loader)

    if is_train:
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders


def build_test_loader(cfg, is_train=False):
    path_catalog = import_file("monoex.config.catalogs", cfg.PATHS_CATALOG, True)
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = samplers.InferenceSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)

    return data_loader


def trivial_batch_collator(batch):
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)
