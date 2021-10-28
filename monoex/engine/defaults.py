import argparse
import os

import torch
from monoex.utils import comm
from monoex.utils.env import collect_env_info, seed_all_rng
from monoex.utils.file_io import PathManager
from monoex.utils.logger import setup_logger

__all__ = ["default_argument_parser", "default_setup"]


def default_argument_parser():
    parser = argparse.ArgumentParser(description="monoex Training")
    parser.add_argument("--config", dest="config_file", default="", metavar="FILE", help="Path to config file")
    parser.add_argument("--eval", dest="eval_only", action="store_true", help="Perform evaluation only")
    parser.add_argument("--eval_iou", action="store_true", help="Evaluate disentangling IoU")
    parser.add_argument("--eval_depth", action="store_true", help="Evaluate depth errors")
    parser.add_argument("--eval_all_depths", action="store_true", help="Evaluate all depth.")
    parser.add_argument("--eval_score_iou", action="store_true", help="Evaluate the relation between score and iou")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--vis", action="store_true", help="Visualize if evaluate")
    parser.add_argument("--ckpt", default=None, help="Path to the checkpoint for test, default is the latest.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpu")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of batch_size")
    parser.add_argument("--num_work", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--output", type=str, default=None, help="Path to save ckpt and log.")
    parser.add_argument("--vis_thres", type=float, default=0.25, help="Threshold for visualize results of detection")
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines for multi-master")
    parser.add_argument("--machine-rank", type=int, default=0, help="The rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    parser.add_argument("--dist-url", default="auto", help="Auto setting for distributed launch.")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    return parser


def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    logger = setup_logger(output_dir, rank, file_name="log_{}.txt".format(cfg.START_TIME))
    logger.info("Use GPU num: {}".format(args.num_gpus))
    logger.info("Use process rank: {}".format(rank))
    logger.info("Use world size: {}".format(comm.get_world_size()))
    logger.info("Use environment info:\n" + collect_env_info())
    logger.info("Command line arguments:\n" + str(args))

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file: {}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read()
            )
        )

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the
    # small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    return logger
