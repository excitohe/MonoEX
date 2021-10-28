import random
import numpy as np
import torch
from PIL import Image

from monoex.config import KITTI_TYPE_ID_CONVERSION
from monoex.structures import ParamList
from monoex.modeling.utils import (
    gaussian_radius, draw_umich_gaussian_ellip, draw_umich_gaussian_round
)

from .kitti_base import KITTIBaseDataset
from .kitti_utils import approx_project_center, convert_rty_to_alpha, refresh_attributes


class KITTIMonoFlexDataset(KITTIBaseDataset):

    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTIMonoFlexDataset,
              self).__init__(cfg=cfg, root=root, is_train=is_train, transforms=transforms)

        self.flip = cfg.INPUT.FLIP.ENABLE
        self.flip_ratio = cfg.INPUT.FLIP.RATIO

        self.filter_anno = cfg.DATASETS.FILTER_ANNO.ENABLE
        self.filter_anno_tru_ratio = cfg.DATASETS.FILTER_ANNO.TRUNC_RATIO
        self.filter_anno_min_boxhw = cfg.DATASETS.FILTER_ANNO.MIN_BOXSIZE

        self.use_objects_outside = cfg.DATASETS.USE_OBJECTS_OUTSIDE
        self.approx_3d_center = cfg.INPUT.APPROX_3D_CENTER
        self.keypoint_visible_modify = cfg.INPUT.KEYPOINT_VISIBLE_MODIFY
        self.orient_mode = cfg.INPUT.ORIENT_MODE
        self.orient_bin_size = cfg.INPUT.ORIENT_BIN_SIZE
        self.heatmap_center = cfg.INPUT.HEATMAP_CENTER
        self.heatmap_ratio = cfg.INPUT.HEATMAP_RATIO
        self.heatmap_adjust = cfg.INPUT.HEATMAP_ADJUST
        self.edge_fusion = cfg.MODEL.HEAD.EDGE_FUSION

        self.max_edge_length = (self.output_w + self.output_h) * 2
        self.alpha_centers = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

    @staticmethod
    def get_edge_utils(image_size, pad_size, down_ratio=4):
        image_w, image_h = image_size
        x_min = np.ceil(pad_size[0] / down_ratio)
        y_min = np.ceil(pad_size[1] / down_ratio)
        x_max = (pad_size[0] + image_w - 1) // down_ratio
        y_max = (pad_size[1] + image_h - 1) // down_ratio
        step = 1
        edge_indices = []
        # left
        y = torch.arange(y_min, y_max, step)
        x = torch.ones(len(y)) * x_min
        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)
        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max
        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)
        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max
        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)
        # top
        x = torch.arange(x_max, x_min - 1, -step)
        y = torch.ones(len(x)) * y_min
        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)
        # concatenate
        edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)
        return edge_indices

    def encode_alpha_multi_bin(self, alpha, num_bin=2, margin=1 / 6):
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin  # pi / 2
        margin_size = bin_size * margin  # pi / 12
        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size
        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi
        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset
        return encode_alpha

    def __getitem__(self, index):
        if index >= self.num_samples:
            index = index % self.num_samples
            image = self.get_image(index, use_right_cam=True)
            calib = self.get_calib(index, use_right_cam=True)
            annos = None if self.split == "test" else self.get_annos(index)
            # update target box2d from right cam
            right_annos = []
            image_w, image_h = image.size
            for obj in annos:
                corner_3d = obj.generate_corner_3d()
                corner_2d, _ = calib.project_rect_to_image(corner_3d)
                obj.box2d = np.array([
                    max(corner_2d[:, 0].min(), 0),
                    max(corner_2d[:, 1].min(), 0),
                    min(corner_2d[:, 0].max(), image_w - 1),
                    min(corner_2d[:, 1].max(), image_h - 1)
                ],
                                     dtype=np.float32)
                obj.xmin, obj.ymin, obj.xmax, obj.ymax = obj.box2d
                right_annos.append(obj)
            annos = right_annos
        else:
            image = self.get_image(index)
            calib = self.get_calib(index)
            annos = None if self.split == "test" else self.get_annos(index)

        origin_index = self.image_files[index][:6]

        # select objects in whitelist
        annos = self.select_whitelist(annos)

        random_flip_flag = False
        if self.is_train and self.flip and random.random() < self.flip_ratio:
            random_flip_flag = True
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            P2 = calib.P.copy()
            P2[0, 2] = image.size[0] - P2[0, 2] - 1
            P2[0, 3] = -P2[0, 3]
            calib.P = P2
            refresh_attributes(calib)

        # pad image
        before_image = np.array(image).copy()
        image_w, image_h = image.size
        image, pad_size = self.pad_image(image)
        origin_image = np.array(image).copy() if self.is_train else before_image

        # the boundaries of the image after padding
        x_min = int(np.ceil(pad_size[0] / self.down_ratio))
        y_min = int(np.ceil(pad_size[1] / self.down_ratio))
        x_max = (pad_size[0] + image_w - 1) // self.down_ratio
        y_max = (pad_size[1] + image_h - 1) // self.down_ratio

        if self.edge_fusion:
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils((image_w, image_h), pad_size).numpy()
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1

        if self.split == "test":
            target = ParamList(image_size=image.size, is_train=self.is_train)
            target.add_field("pad_size", pad_size)
            target.add_field("calib", calib)
            target.add_field("origin_image", origin_image)
            if self.edge_fusion:
                target.add_field('edge_len', input_edge_count)
                target.add_field('edge_ind', input_edge_indices)
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target, origin_index

        tgt_hmp2ds = np.zeros([self.num_classes, self.output_h, self.output_w], dtype=np.float32)
        tgt_clsids = np.zeros([self.max_objs], dtype=np.int32)
        tgt_loc3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_dim3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_rty3ds = np.zeros([self.max_objs], dtype=np.float32)
        tgt_alphas = np.zeros([self.max_objs], dtype=np.float32)

        tgt_box2ds = np.zeros([self.max_objs, 4], dtype=np.float32)
        tgt_boxgts = np.zeros([self.max_objs, 4], dtype=np.float32)
        tgt_kpt2ds = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
        tgt_occlus = np.zeros([self.max_objs], dtype=np.float32)
        tgt_truncs = np.zeros([self.max_objs], dtype=np.float32)
        tgt_offset_3ds = np.zeros([self.max_objs, 2], dtype=np.float32)
        tgt_center_2ds = np.zeros([self.max_objs, 2], dtype=np.int32)

        tgt_weight_3ds = np.zeros([self.max_objs], dtype=np.float32)
        tgt_kpt_depth_mask = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_fill_mask = np.zeros([self.max_objs], dtype=np.uint8)
        tgt_flip_mask = np.zeros([self.max_objs], dtype=np.uint8)
        tgt_trun_mask = np.zeros([self.max_objs], dtype=np.uint8)

        if self.orient_mode == 'head-axis':
            tgt_orient_3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        else:
            tgt_orient_3ds = np.zeros([self.max_objs, self.orient_bin_size * 2], dtype=np.float32)

        for i, obj in enumerate(annos):
            cls = obj.type
            cls_id = KITTI_TYPE_ID_CONVERSION[cls]
            if cls_id < 0:
                continue

            if random_flip_flag:
                obj_w = obj.xmax - obj.xmin
                obj.xmin = image_w - obj.xmax - 1
                obj.xmax = obj.xmin + obj_w
                obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)

                rty = obj.rty
                rty = (-np.pi - rty) if rty < 0 else (np.pi - rty)
                while rty > np.pi:
                    rty -= np.pi * 2
                while rty < (-np.pi):
                    rty += np.pi * 2
                obj.rty = rty

                loc = obj.t.copy()
                loc[0] = -loc[0]
                obj.t = loc

                obj.alpha = convert_rty_to_alpha(rty, obj.t[2], obj.t[0])

            # convert location_bottom to location_center
            loc = obj.t.copy()
            loc[1] = loc[1] - obj.h / 2
            if loc[-1] <= 0:
                continue

            corner_3d = obj.generate_corner_3d()
            corner_2d, _ = calib.project_rect_to_image(corner_3d)
            project_box_2d = np.array([
                corner_2d[:, 0].min(),
                corner_2d[:, 1].min(),
                corner_2d[:, 0].max(),
                corner_2d[:, 1].max(),
            ])

            if (
                project_box_2d[0] >= 0 and project_box_2d[2] <= image_w - 1
                and project_box_2d[1] >= 0 and project_box_2d[3] <= image_h - 1
            ):
                box_2d = project_box_2d.copy()
            else:
                box_2d = obj.box2d.copy()

            if self.filter_anno:
                if (
                    self.filter_anno_tru_ratio >= obj.truncation and self.filter_anno_min_boxhw >=
                    (box_2d[2:] - box_2d[:2]).min()
                ):
                    continue

            project_center, _ = calib.project_rect_to_image(loc.reshape(-1, 3))
            project_center = project_center[0]
            project_inside_image = ((0 <= project_center[0] <= image_w - 1) &
                                    (0 <= project_center[1] <= image_h - 1))

            approx_center_flag = False
            if not project_inside_image:
                if self.use_objects_outside:
                    approx_center_flag = True
                    center_2d = (box_2d[:2] + box_2d[2:]) / 2
                    if self.approx_3d_center == 'intersect':
                        target_project_center, _ = approx_project_center(
                            project_center, center_2d.reshape(1, 2), (image_w, image_h)
                        )
                    else:
                        raise NotImplementedError
                else:
                    continue
            else:
                target_project_center = project_center.copy()

            center_bt = np.stack((corner_3d[:4].mean(axis=0), corner_3d[4:].mean(axis=0)), axis=0)
            kpt_3d = np.concatenate((corner_3d, center_bt), axis=0)
            kpt_2d, _ = calib.project_rect_to_image(kpt_3d)

            # keypoint must be inside the image and in front of the camera
            kpt_x_vis = (kpt_2d[:, 0] >= 0) & (kpt_2d[:, 0] <= image_w - 1)
            kpt_y_vis = (kpt_2d[:, 1] >= 0) & (kpt_2d[:, 1] <= image_h - 1)
            kpt_z_vis = (kpt_3d[:, -1] > 0)

            # xyz visible
            kpt_vis = kpt_x_vis & kpt_y_vis & kpt_z_vis
            # center, diag-02, diag-13
            kpt_depth_valid = np.stack((
                kpt_vis[[8, 9]].all(),
                kpt_vis[[0, 2, 4, 6]].all(),
                kpt_vis[[1, 3, 5, 7]].all(),
            ))

            if self.keypoint_visible_modify:
                kpt_vis = np.append(
                    np.tile(kpt_vis[:4] | kpt_vis[4:8], 2),
                    np.tile(kpt_vis[8] | kpt_vis[9], 2),
                )
                kpt_depth_valid = np.stack((
                    kpt_vis[[8, 9]].all(),
                    kpt_vis[[0, 2, 4, 6]].all(),
                    kpt_vis[[1, 3, 5, 7]].all(),
                ))
                kpt_vis = kpt_vis.astype(np.float32)
                kpt_depth_valid = kpt_depth_valid.astype(np.float32)

            kpt_2d = (kpt_2d + pad_size.reshape(1, 2)) / self.down_ratio
            target_project_center = (target_project_center + pad_size) / self.down_ratio
            project_center = (project_center + pad_size) / self.down_ratio

            box_2d[0::2] += pad_size[0]
            box_2d[1::2] += pad_size[1]
            box_2d /= self.down_ratio

            # box_2d_center & box_2d_size
            box_2d_center = (box_2d[:2] + box_2d[2:]) / 2
            box_2d_size = box_2d[2:] - box_2d[:2]

            if self.heatmap_center == '2D':
                target_center = box_2d_center.round().astype(np.int)
            else:
                target_center = target_project_center.round().astype(np.int)

            # clip to the boundary
            target_center[0] = np.clip(target_center[0], x_min, x_max)
            target_center[1] = np.clip(target_center[1], y_min, y_max)

            pred_2d = True
            if not (
                target_center[0] >= box_2d[0] and target_center[1] >= box_2d[1]
                and target_center[0] <= box_2d[2] and target_center[1] <= box_2d[3]
            ):
                pred_2d = False

            if ((box_2d_size > 0).all() and (0 <= target_center[0] <= self.output_w - 1)
                and (0 <= target_center[1] <= self.output_h - 1)):
                rty = obj.rty
                alpha = obj.alpha

                if self.heatmap_adjust and approx_center_flag:
                    # for outside objects, generate 1-dim heatmap
                    box_w = min(target_center[0] - box_2d[0], box_2d[2] - target_center[0])
                    box_h = min(target_center[1] - box_2d[1], box_2d[3] - target_center[1])
                    radius_x = box_w * self.heatmap_ratio
                    radius_y = box_h * self.heatmap_ratio
                    radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    tgt_hmp2ds[cls_id] = draw_umich_gaussian_ellip(
                        tgt_hmp2ds[cls_id], target_center, radius_x, radius_y
                    )
                else:
                    # for inside objects, generate circular heatmap
                    radius = gaussian_radius(box_2d_size[1], box_2d_size[0])
                    radius = max(0, int(radius))
                    tgt_hmp2ds[cls_id] = draw_umich_gaussian_round(
                        tgt_hmp2ds[cls_id], target_center, radius
                    )

                tgt_clsids[i] = cls_id
                tgt_loc3ds[i] = loc
                tgt_dim3ds[i] = np.array([obj.l, obj.h, obj.w], dtype=np.float32)
                tgt_rty3ds[i] = rty
                tgt_alphas[i] = alpha
                tgt_boxgts[i] = obj.box2d.copy()  # for visualization
                if pred_2d:
                    tgt_box2ds[i] = box_2d
                tgt_kpt2ds[i] = np.concatenate(
                    (kpt_2d - target_center.reshape(1, -1), kpt_vis[:, np.newaxis]), axis=1
                )
                tgt_orient_3ds[i] = self.encode_alpha_multi_bin(alpha, num_bin=self.orient_bin_size)
                tgt_center_2ds[i] = target_center
                tgt_offset_3ds[i] = project_center - target_center
                tgt_occlus[i] = obj.occlusion
                tgt_truncs[i] = obj.truncation
                tgt_weight_3ds[i] = 1
                tgt_kpt_depth_mask[i] = kpt_depth_valid
                tgt_fill_mask[i] = 1
                tgt_trun_mask[i] = int(approx_center_flag)
                tgt_flip_mask[i] = 1 if random_flip_flag else 0

        target = ParamList(image_size=image.size, is_train=self.is_train)
        target.add_field("tgt_hmp2ds", tgt_hmp2ds)
        target.add_field("tgt_clsids", tgt_clsids)
        target.add_field("tgt_loc3ds", tgt_loc3ds)
        target.add_field("tgt_dim3ds", tgt_dim3ds)
        target.add_field("tgt_rty3ds", tgt_rty3ds)
        target.add_field("tgt_box2ds", tgt_box2ds)
        target.add_field("tgt_boxgts", tgt_boxgts)
        target.add_field("tgt_kpt2ds", tgt_kpt2ds)
        target.add_field("tgt_alphas", tgt_alphas)
        target.add_field("tgt_occlus", tgt_occlus)
        target.add_field("tgt_truncs", tgt_truncs)
        target.add_field("tgt_orient_3ds", tgt_orient_3ds)
        target.add_field("tgt_center_2ds", tgt_center_2ds)
        target.add_field("tgt_offset_3ds", tgt_offset_3ds)
        target.add_field("tgt_weight_3ds", tgt_weight_3ds)
        target.add_field("tgt_kpt_depth_mask", tgt_kpt_depth_mask)
        target.add_field("tgt_flip_mask", tgt_flip_mask)
        target.add_field("tgt_fill_mask", tgt_fill_mask)
        target.add_field("tgt_trun_mask", tgt_trun_mask)
        target.add_field("calib", calib)
        target.add_field("pad_size", pad_size)
        target.add_field("origin_image", origin_image)

        if self.edge_fusion:
            target.add_field('edge_len', input_edge_count)
            target.add_field('edge_ind', input_edge_indices)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, origin_index
