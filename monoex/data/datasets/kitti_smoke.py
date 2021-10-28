import random

import numpy as np
from PIL import Image
from monoex.config import KITTI_TYPE_ID_CONVERSION
from monoex.modeling.utils import (
    affine_transform, draw_umich_gaussian_round, gaussian_radius, get_transfrom_matrix
)
from monoex.structures import ParamList

from .kitti_base import KITTIBaseDataset
from .kitti_utils import refresh_attributes


class KITTISMOKEDataset(KITTIBaseDataset):

    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTISMOKEDataset,
              self).__init__(cfg=cfg, root=root, is_train=is_train, transforms=transforms)

        self.flip = cfg.INPUT.FLIP.ENABLE
        self.flip_ratio = cfg.INPUT.FLIP.RATIO

        self.affine = cfg.INPUT.AFFINE.ENABLE
        self.affine_ratio = cfg.INPUT.AFFINE.RATIO
        self.affine_shift = cfg.INPUT.AFFINE.SHIFT
        self.affine_scale = cfg.INPUT.AFFINE.SCALE

    def __len__(self):
        return self.num_samples

    @staticmethod
    def encode_label(K, ry, dims, locs):
        l, h, w = dims[0], dims[1], dims[2]
        x, y, z = locs[0], locs[1], locs[2]
        x_corners = [0, l, l, l, l, 0, 0, 0]
        y_corners = [0, 0, h, h, 0, 0, h, h]
        z_corners = [0, 0, 0, w, w, w, w, 0]
        x_corners += -np.float32(l) / 2
        y_corners += -np.float32(h)
        z_corners += -np.float32(w) / 2
        corners_3d = np.array([x_corners, y_corners, z_corners])
        rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        corners_3d = np.matmul(rot_mat, corners_3d)
        corners_3d += np.array([x, y, z]).reshape([3, 1])
        loc_center = np.array([x, y - h / 2, z])
        proj_point = np.matmul(K, loc_center)
        proj_point = proj_point[:2] / proj_point[2]
        corners_2d = np.matmul(K, corners_3d)
        corners_2d = corners_2d[:2] / corners_2d[2]
        box2d = np.array([
            min(corners_2d[0]),
            min(corners_2d[1]),
            max(corners_2d[0]),
            max(corners_2d[1])
        ])
        return proj_point, box2d, corners_3d

    def __getitem__(self, index):
        image = self.get_image(index)
        calib = self.get_calib(index)
        annos = None if self.split == "test" else self.get_annos(index)

        origin_index = self.image_files[index][:6]

        origin_size = np.array(image.size, dtype=np.float32)
        center = origin_size / 2

        random_flip_flag = False
        if self.is_train and self.flip and random.random() < self.flip_ratio:
            random_flip_flag = True
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            P2 = calib.P.copy()
            P2[0, 2] = origin_size[0] - center[0] - 1
            calib.P = P2
            refresh_attributes(calib)

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

        trans_inp = get_transfrom_matrix(center_size, [self.input_w, self.input_h])
        trans_inp_inv = np.linalg.inv(trans_inp)
        trans_out = get_transfrom_matrix(center_size, [self.output_w, self.output_h])

        image = image.transform(
            size=(self.input_w, self.input_h),
            method=Image.AFFINE,
            data=trans_inp_inv.flatten()[:6],
            resample=Image.BILINEAR
        )

        if not self.is_train:
            target = ParamList(image_size=origin_size, is_train=self.is_train)
            target.add_field("trans_mat", trans_out)
            target.add_field("K", calib.P[:3, :3].astype(np.float32))
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target, origin_index

        tgt_hmp2ds = np.zeros([self.num_classes, self.output_h, self.output_w], dtype=np.float32)
        tgt_clsids = np.zeros([self.max_objs], dtype=np.int32)
        tgt_dim3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_loc3ds = np.zeros([self.max_objs, 3], dtype=np.float32)
        tgt_rty3ds = np.zeros([self.max_objs], dtype=np.float32)

        tgt_corner_3ds = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        tgt_center_2ds = np.zeros([self.max_objs, 2], dtype=np.int32)
        tgt_fill_mask = np.zeros([self.max_objs], dtype=np.uint8)
        tgt_flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        for i, obj in enumerate(annos):
            cls = obj.type
            cls_id = KITTI_TYPE_ID_CONVERSION[cls]
            if cls_id < 0:
                continue

            loc_b = obj.t.copy()  # location_bottom
            rty = obj.rty
            if random_flip_flag:
                loc_b[0] *= -1
                rty *= -1
            """
            dim = np.array([obj.l, obj.h, obj.w], dtype=np.float32)
            point, box2d, box3d = self.encode_label(calib.P, rty, dim, loc_b) 
            """

            corner_3d = obj.generate_corner_3d()
            corner_2d, _ = calib.project_rect_to_image(corner_3d)
            project_box_2d = np.array([
                corner_2d[:, 0].min(),
                corner_2d[:, 1].min(),
                corner_2d[:, 0].max(),
                corner_2d[:, 1].max(),
            ])

            loc_c = loc_b.copy()
            loc_c[1] = loc_c[1] - obj.h / 2  # location_center
            project_center, _ = calib.project_rect_to_image(loc_c.reshape(-1, 3))
            project_center = project_center[0]

            project_center = affine_transform(project_center, trans_out)
            project_box_2d[:2] = affine_transform(project_box_2d[:2], trans_out)
            project_box_2d[2:] = affine_transform(project_box_2d[2:], trans_out)
            project_box_2d[0::2] = project_box_2d[0::2].clip(0, self.output_w - 1)
            project_box_2d[1::2] = project_box_2d[1::2].clip(0, self.output_h - 1)
            box_h = project_box_2d[3] - project_box_2d[1]
            box_w = project_box_2d[2] - project_box_2d[0]

            if ((0 < project_center[0] < self.output_w)
                and (0 < project_center[1] < self.output_h)):
                point_int = project_center.astype(np.int32)
                radius = gaussian_radius(box_h, box_w)
                radius = max(0, int(radius))
                tgt_hmp2ds[cls_id] = draw_umich_gaussian_round(
                    tgt_hmp2ds[cls_id], project_center, radius
                )
                tgt_clsids[i] = cls_id
                tgt_dim3ds[i] = np.array([obj.l, obj.h, obj.w])
                tgt_loc3ds[i] = loc_b
                tgt_rty3ds[i] = rty
                tgt_center_2ds[i] = point_int
                tgt_corner_3ds[i] = corner_3d.transpose(1, 0)[:, [6, 5, 1, 0, 4, 7, 3, 2]]
                tgt_fill_mask[i] = 1 if not random_affine_flag else 0
                tgt_flip_mask[i] = 1 if not random_affine_flag and random_flip_flag else 0

        target = ParamList(image_size=image.size, is_train=self.is_train)
        target.add_field("tgt_clsids", tgt_clsids)
        target.add_field("tgt_hmp2ds", tgt_hmp2ds)
        target.add_field("tgt_loc3ds", tgt_loc3ds)
        target.add_field("tgt_dim3ds", tgt_dim3ds)
        target.add_field("tgt_rty3ds", tgt_rty3ds)
        target.add_field("tgt_center_2ds", tgt_center_2ds)
        target.add_field("tgt_corner_3ds", tgt_corner_3ds)
        target.add_field("trans_mat", trans_out)
        target.add_field("K", calib.P[:3, :3].astype(np.float32))
        target.add_field("tgt_fill_mask", tgt_fill_mask)
        target.add_field("tgt_flip_mask", tgt_flip_mask)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, origin_index
