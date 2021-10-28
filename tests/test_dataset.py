import cv2
import ipdb
import torch
import torch.utils.data
from monoex.config import get_cfg
from monoex.data import build_dataset
from monoex.data.collate import BatchCollator
from monoex.data.transforms import build_transforms
from monoex.engine import default_argument_parser
from monoex.structures.image_list import to_image_list
from monoex.utils.env import import_file


def test_iteration(data_loader):
    for idx, data in enumerate(data_loader):
        images = data["images"]
        targets = [target for target in data["targets"]]
        imgids = data["img_ids"]

        print(f'Index: {idx} | ImgID: {imgids}')

        images = to_image_list(images)

        # heatmap = torch.stack([t.get_field("heatmap") for t in targets])
        # size_2d = torch.stack([t.get_field("size_2d") for t in targets])
        # offset_2d = torch.stack([t.get_field("offset_2d") for t in targets])
        # depth = torch.stack([t.get_field("depth") for t in targets])
        # heading_bin = torch.stack([t.get_field("heading_bin") for t in targets])
        # heading_res = torch.stack([t.get_field("heading_res") for t in targets])
        # src_size_3d = torch.stack([t.get_field("src_size_3d") for t in targets])
        # size_3d = torch.stack([t.get_field("size_3d") for t in targets])
        # offset_3d = torch.stack([t.get_field("offset_3d") for t in targets])
        # cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
        # indices = torch.stack([t.get_field("indices") for t in targets])
        # mask_2d = torch.stack([t.get_field("mask_2d") for t in targets])
        # K = torch.stack([t.get_field("K") for t in targets])
        # coord_range = torch.stack([t.get_field("coord_range") for t in targets])
        # down_ratio = torch.stack([t.get_field("down_ratio") for t in targets])

        # hmp = heatmap.numpy()[0].transpose([1, 2, 0]) * 255.0
        # hmp = hmp[:,:,[1,0,2]]
        # # cv2.imwrite('hmp.png', hmp)

        ipdb.set_trace()


def test_build_data_loader(cfg, is_train=True):
    path_catalog = import_file("monoex.config.catalogs", cfg.PATHS_CATALOG, True)
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

    data_loader = torch.utils.data.DataLoader(datasets[0], batch_size=1, collate_fn=collator)
    return data_loader


def setup(args):
    # fetch config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # modify special parameters
    cfg.DATASETS.FILTER_ANNO.ENABLE = False
    cfg.INPUT.FLIP.ENABLE = False
    cfg.INPUT.AFFINE.ENABLE = False

    return cfg


def main():
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    data_loader = test_build_data_loader(cfg, is_train=True)
    test_iteration(data_loader)


if __name__ == '__main__':
    main()
