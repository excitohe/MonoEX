import os

from monoex.data import build_test_loader
from monoex.engine.infer_api import inference, inference_all_depths
from monoex.utils import comm
from monoex.utils.file_io import PathManager


def do_test(cfg, model, vis, eval_score_iou, eval_all_depths=True):
    eval_types = ("detection", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            PathManager.mkdirs(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = build_test_loader(cfg)
    inference_func = inference_all_depths if eval_all_depths else inference

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result_dict, result_str, dis_ious = inference_func(
            model,
            data_loaders_val,
            dataset_name=dataset_name,
            eval_types=eval_types,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
            metrics=cfg.TEST.METRIC,
            vis=vis,
            eval_score_iou=eval_score_iou,
        )
        comm.synchronize()
