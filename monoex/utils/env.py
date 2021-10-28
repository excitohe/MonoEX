import datetime
import logging
import os
import random
import sys
from importlib import util

import numpy as np
import torch
from torch.utils.collect_env import get_pretty_env_info

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() + int(datetime.datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big"))
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_env_info():
    env_str = get_pretty_env_info()
    return env_str


def import_file(module_name, file_path, make_importable=False):
    spec = util.spec_from_file_location(module_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if make_importable:
        sys.modules[module_name] = module_name
    return module
