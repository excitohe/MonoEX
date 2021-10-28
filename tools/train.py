import datetime
import torch

from monoex.config import get_cfg
from monoex.data import build_data_loader, build_test_loader
from monoex.engine import default_argument_parser, default_setup, launch
from monoex.engine.test_api import do_test
from monoex.engine.train_api import do_train
from monoex.modeling.detectors import build_detector
from monoex.solver import build_optimizer, build_scheduler
from monoex.utils import comm
from monoex.utils.checkpoint import DetectronCheckpointer


def train(cfg, model, device, distributed, logger):
    data_loader = build_data_loader(cfg, is_train=True)
    test_loader = build_test_loader(cfg, is_train=False)

    # use `epoch` rather `iteration` for save and validate checkpoint
    total_iters_each_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
        cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * total_iters_each_epoch
        cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = total_iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
        cfg.SOLVER.EVAL_INTERVAL = total_iters_each_epoch * cfg.SOLVER.EVAL_EPOCH_INTERVAL
        cfg.SOLVER.STEPS = [total_iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
        cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * total_iters_each_epoch

    cfg.freeze()
    logger.info("Running with config:\n{}".format(cfg))

    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_scheduler = build_scheduler(optimizer, cfg=cfg)

    arguments = {}
    arguments['iteration'] = 0
    arguments['iter_per_epoch'] = total_iters_each_epoch

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = comm.get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)

    if len(cfg.MODEL.WEIGHT) > 0:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False)
        arguments.update(extra_checkpoint_data)

    do_train(
        cfg,
        distributed,
        model,
        data_loader,
        test_loader,
        optimizer,
        scheduler,
        warmup_scheduler,
        checkpointer,
        device,
        arguments,
    )


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth

    if args.vis_thres > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thres
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output
    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test", )

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d_%H:%M:%S')
    logger = default_setup(cfg, args)
    return cfg, logger


def main(args):
    cfg, logger = setup(args)

    distributed = comm.get_world_size() > 1
    if not distributed:
        cfg.MODEL.USE_SYNC_BN = False

    model = build_detector(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.eval_only:
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

        return do_test(
            cfg,
            checkpointer.model,
            vis=args.vis,
            eval_score_iou=args.eval_score_iou,
            eval_all_depths=args.eval_all_depths,
        )

    if distributed:
        # convert BN to SyncBN
        if cfg.MODEL.USE_SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    train(cfg, model, device, distributed, logger)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
