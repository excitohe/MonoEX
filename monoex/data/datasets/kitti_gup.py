import random

import numpy as np
from PIL import Image
from monoex.config import KITTI_TYPE_ID_CONVERSION
from monoex.modeling.utils import (
    affine_transform, draw_umich_gaussian_round, gaussian_radius, get_transfrom_matrix
)
from monoex.structures import ParamList

from .kitti_base import KITTIBaseDataset
from .kitti_utils import angle_to_class, refresh_attributes


class KITTIGUPDataset(KITTIBaseDataset):

    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTIGUPDataset,
              self).__init__(cfg=cfg, root=root, is_train=is_train, transforms=transforms)

        self.dim_3d_mean = cfg.HEAD.DIM_MEAN
        # self.dim_3d_mean = (
        #     (3.88311640, 1.52563191, 1.62856740),
        #     (0.84422524, 1.76255119, 0.66068622),
        #     (1.76282397, 1.73698127, 0.59706367),
        # )

        self.flip = cfg.INPUT.FLIP.ENABLE
        self.flip_ratio = cfg.INPUT.FLIP.RATIO

        self.affine = cfg.INPUT.AFFINE.ENABLE
        self.affine_ratio = cfg.INPUT.AFFINE.RATIO
        self.affine_shift = cfg.INPUT.AFFINE.SHIFT
        self.affine_scale = cfg.INPUT.AFFINE.SCALE

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.get_image(index)
        calib = self.get_calib(index)
        annos = None if self.split == "test" else self.get_annos(index)

        origin_index = self.image_files[index][:6]

        annos = self.select_whitelist(annos)

        origin_size = np.array(image.size, dtype=np.float32)
        center = np.array(origin_size) / 2
        affine_size = origin_size

        size = np.array([i for i in image.size], dtype=np.float32)
        center = size / 2

        random_flip_flag = False
        if self.is_train and self.flip and random.random() < self.flip_ratio:
            random_flip_flag = True
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            calib.flip(origin_size)

        random_affine_flag = False
        if self.is_train and self.affine and random.random() < self.affine_ratio:
            random_affine_flag = True
            affine_size = origin_size * np.clip(
                np.random.randn() * self.affine_scale + 1, 1 - self.affine_scale,
                1 + self.affine_scale
            )
            center[0] += origin_size[0] * np.clip(
                np.random.randn() * self.affine_shift, -2 * self.affine_shift, 2 * self.affine_shift
            )
            center[1] += origin_size[1] * np.clip(
                np.random.randn() * self.affine_shift, -2 * self.affine_shift, 2 * self.affine_shift
            )

        center_size = [center, affine_size]
        trans_input = get_transfrom_matrix(center_size, [self.input_w, self.input_h])
        trans_input_inv = np.linalg.inv(trans_input)

        image = image.transform((self.input_w, self.input_h),
                                method=Image.AFFINE,
                                data=trans_input_inv.flatten()[:6],
                                resample=Image.BILINEAR)

        coord_range = np.array([center - affine_size / 2,
                                center + affine_size / 2]).astype(np.float32)

        trans_mat = get_transfrom_matrix(center_size, [self.output_w, self.output_h])

        if self.split == 'test':
            target = ParamList(image_size=origin_size, is_train=self.is_train)
            target.add_field("K", calib.P.astype(np.float32))
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target, origin_index

        tgt_hmp2ds = np.zeros([self.num_classes, self.output_h, self.output_w], dtype=np.float32)
        tgt_clsids = np.zeros([self.max_objs], dtype=np.int32)
        tgt_dim2ds = np.zeros([self.max_objs, 2], dtype=np.float32)
        tgt_src_dim_3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_dim3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_dep3ds = np.zeros([self.max_objs, 1], dtype=np.float32)
        tgt_head_bin = np.zeros([self.max_objs, 1], dtype=np.int32)
        tgt_head_res = np.zeros([self.max_objs, 1], dtype=np.float32)
        tgt_offset_2ds = np.zeros([self.max_objs, 2], dtype=np.float32)
        tgt_offset_3ds = np.zeros([self.max_objs, 2], dtype=np.float32)
        tgt_indices = np.zeros([self.max_objs], dtype=np.int32)
        tgt_mask = np.zeros([self.max_objs], dtype=np.uint8)

        for i, obj in enumerate(annos):
            cls = obj.type
            cls_id = KITTI_TYPE_ID_CONVERSION[cls]
            if cls_id < 0:
                continue

            # filter inappropriate samples by difficulty
            if obj.level_str == "UnKnown" or obj.t[-1] < 2:
                continue

            if random_flip_flag:
                box_w = obj.xmax - obj.xmin
                obj.xmin = size[0] - obj.xmax - 1
                obj.xmax = obj.xmin + box_w
                obj.box2d = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)

                obj.t[0] *= -1
                # FIXME: GUP and MonoFlex, which is better?
                rty = obj.rty
                rty = np.pi - rty
                if rty > np.pi:
                    rty -= 2 * np.pi
                if rty < -np.pi:
                    rty += 2 * np.pi
                obj.rty = rty

            loc_c = obj.t.copy()  # actual location_bottom
            loc_c[1] = loc_c[1] - obj.h / 2  # now is location_center

            # process box2d & get center_2d
            # FIXME: Replace with project_box2d can gain better?
            box_2d = obj.box2d.copy()
            box_2d[:2] = affine_transform(box_2d[:2], trans_mat)
            box_2d[2:] = affine_transform(box_2d[2:], trans_mat)

            # process box3d & get center_3d
            center_2d = np.array([
                (box_2d[0] + box_2d[2]) / 2,
                (box_2d[1] + box_2d[3]) / 2,
            ],
                                 dtype=np.float32)
            center_3d, _ = calib.project_rect_to_image(loc_c.reshape(-1, 3))
            center_3d = center_3d[0]

            center_3d = affine_transform(center_3d.reshape(-1), trans_mat)

            # generate the center of gaussian tgt_hmp2ds [optional: 3d center or 2d center]
            target_center = center_3d.astype(np.int32)
            if target_center[0] < 0 or target_center[0] > self.output_w:
                continue
            if target_center[1] < 0 or target_center[1] > self.output_h:
                continue

            # generate the radius of gaussian tgt_hmp2ds
            box_w, box_h = box_2d[2] - box_2d[0], box_2d[3] - box_2d[1]
            radius = gaussian_radius(box_w, box_h)  # TODO: switch order
            # radius = gaussian_radius(box_h, box_w)
            radius = max(0, int(radius))

            if obj.type in ['Van', 'Truck', 'DontCare']:
                draw_umich_gaussian_round(tgt_hmp2ds[0], target_center, radius)

            draw_umich_gaussian_round(tgt_hmp2ds[cls_id], target_center, radius)

            tgt_clsids[i] = cls_id

            # encoding 2d/3d offset & 2d size
            tgt_indices[i] = target_center[1] * self.output_w + target_center[0]
            tgt_offset_2ds[i] = center_2d - target_center
            tgt_dim2ds[i] = box_w * 1.0, box_h * 1.0

            # encoding tgt_dep3ds
            tgt_dep3ds[i] = obj.t[-1]

            # encoding heading angle
            head_angle = calib.ry2alpha(obj.rty, (obj.box2d[0] + obj.box2d[2]) / 2)
            if head_angle > np.pi:
                head_angle -= 2 * np.pi
            if head_angle < -np.pi:
                head_angle += 2 * np.pi
            tgt_head_bin[i], tgt_head_res[i] = angle_to_class(head_angle)

            # encoding 3d offset & tgt_dim3ds
            tgt_offset_3ds[i] = center_3d - target_center
            tgt_src_dim_3ds[i] = np.array([obj.l, obj.h, obj.w], dtype=np.float32)
            tgt_dim3ds[i] = tgt_src_dim_3ds[i] - self.dim_3d_mean[cls_id]

            if obj.truncation <= 0.5 and obj.occlusion <= 2:
                tgt_mask[i] = 1

        target = ParamList(image_size=image.size, is_train=self.is_train)
        target.add_field("tgt_hmp2ds", tgt_hmp2ds)
        target.add_field("tgt_clsids", tgt_clsids)
        target.add_field("tgt_dim3ds", tgt_dim3ds)
        target.add_field("tgt_dim2ds", tgt_dim2ds)
        target.add_field("tgt_dep3ds", tgt_dep3ds)
        target.add_field("tgt_head_bin", tgt_head_bin)
        target.add_field("tgt_head_res", tgt_head_res)
        target.add_field("tgt_src_dim_3ds", tgt_src_dim_3ds)
        target.add_field("tgt_offset_2ds", tgt_offset_2ds)
        target.add_field("tgt_offset_3ds", tgt_offset_3ds)
        target.add_field("tgt_indices", tgt_indices)
        target.add_field("tgt_mask", tgt_mask)
        target.add_field("K", calib.P.astype(np.float32))
        target.add_field("coord_range", coord_range)
        target.add_field("down_ratio", 4)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, origin_index
