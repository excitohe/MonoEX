import numpy as np
import torch
import torch.nn.functional as F


class MonoFlexCoder(object):

    def __init__(self, cfg):
        super(MonoFlexCoder, self).__init__()

        device = cfg.MODEL.DEVICE
        self.INF = 100000000
        self.EPS = 1e-3

        self.num_class = len(cfg.DATASETS.CLASS_NAMES)
        self.min_radius = cfg.DATASETS.MIN_RADIUS
        self.max_radius = cfg.DATASETS.MAX_RADIUS
        self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
        self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
        self.center_mode = cfg.MODEL.HEAD.CENTER_MODE  # ['max', 'area']

        self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
        self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
        self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REF).to(device=device)

        self.dim_mode = cfg.MODEL.HEAD.DIM_MODE
        self.dim_mean = torch.as_tensor(cfg.MODEL.HEAD.DIM_MEAN).to(device=device)
        self.dim_stds = torch.as_tensor(cfg.MODEL.HEAD.DIM_STDS).to(device=device)

        self.alpha_centers = torch.tensor([0, np.pi / 2, np.pi, -np.pi / 2]).to(device=device)
        self.orient_mode = cfg.INPUT.ORIENT_MODE
        self.orient_bin_size = cfg.INPUT.ORIENT_BIN_SIZE

        self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
        self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_h = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_w = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.K = self.output_w * self.output_h

    @staticmethod
    def rad_to_matrix(rtys, N):
        device = rtys.device
        cos, sin = rtys.cos(), rtys.sin()
        tmp = torch.tensor([[1, 0, 1], [0, 1, 0], [-1, 0, 1]]).to(dtype=torch.float32, device=device)
        mat = tmp.repeat(N, 1).view(N, -1, 3)
        mat[:, 0, 0] *= cos
        mat[:, 0, 2] *= sin
        mat[:, 2, 0] *= sin
        mat[:, 2, 2] *= cos
        return mat

    def encode_box3d(self, rtys, dims, locs):
        if len(rtys.shape) == 2:
            rtys = rtys.flatten()
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)
        device = rtys.device
        N = rtys.shape[0]
        mat = self.rad_to_matrix(rtys, N)
        # l, h, w
        dims_corners = dims.view(-1, 1).repeat(1, 8)
        dims_corners = dims_corners * 0.5
        dims_corners[:, 4:] = -dims_corners[:, 4:]
        index = torch.tensor([
            [4, 5, 0, 1, 6, 7, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 0, 1, 5, 6, 2, 3, 7],
        ]).repeat(N, 1).to(device=device)
        box_3d_object = torch.gather(dims_corners, 1, index)
        box_3d = torch.matmul(mat, box_3d_object.view(N, 3, -1))
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)
        return box_3d.permute(0, 2, 1)

    def decode_box2d_fcos(self, centers, offsets, pad_size=None, out_size=None):
        box2d_center = centers.view(-1, 2)
        box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
        # left, top, right, bottom
        box2d[:, :2] = box2d_center - offsets[:, :2]
        box2d[:, 2:] = box2d_center + offsets[:, 2:]
        if pad_size is not None:
            N = box2d.shape[0]
            out_size = out_size[0]
            box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
            box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
            box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)
        return box2d

    def decode_depth(self, offsets):
        """
        :param: offsets: transform depth_offset to depth
        """
        if self.depth_mode == 'exp':
            depth = offsets.exp()
        elif self.depth_mode == 'linear':
            depth = offsets * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(offsets) - 1
        else:
            raise ValueError(f"Unsupport depth decode mode {self.depth_mode}")
        if self.depth_range is not None:
            depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])
        return depth

    def decode_loc3d(self, points, offsets, depths, calibs, pad_size, batch_idxs):
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        locations = points.new_zeros(points.shape[0], 3).float()
        points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]
        for idx, gt in enumerate(gts):
            corr_pts_idx = torch.nonzero(batch_idxs == gt, as_tuple=False).squeeze(-1)
            calib = calibs[gt]
            # concatenate uv with depth
            corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
            locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)
        return locations

    def decode_depth_from_kpts(self, pred_offsets, pred_kpts, pred_dims, calibs, average_center=False):
        # pred_kpts: [K,10,2], 8 vertices, bottom center and top center
        assert len(calibs) == 1  # for inference, batch size is always 1

        calib = calibs[0]
        # we only need the values of y
        pred_h3d = pred_dims[:, 1]
        pred_kpts = pred_kpts.view(-1, 10, 2)
        # center height -> depth
        if average_center:
            update_pred_keypoints = pred_kpts - pred_offsets.view(-1, 1, 2)
            center_height = update_pred_keypoints[:, -2:, 1]
            center_depth = calib.f_u * pred_h3d.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
            center_depth = center_depth.mean(dim=1)
        else:
            center_height = pred_kpts[:, -2, 1] - pred_kpts[:, -1, 1]
            center_depth = calib.f_u * pred_h3d / (center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_kpts[:, [0, 2], 1] - pred_kpts[:, [4, 6], 1]
        corner_13_height = pred_kpts[:, [1, 3], 1] - pred_kpts[:, [5, 7], 1]
        corner_02_depth = calib.f_u * pred_h3d.unsqueeze(-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib.f_u * pred_h3d.unsqueeze(-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = corner_02_depth.mean(dim=1)
        corner_13_depth = corner_13_depth.mean(dim=1)
        # K x 3
        pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)
        return pred_depths

    def decode_depth_from_kpts_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        pred_h3d = pred_dimensions[:, 1].clone()
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
            calib = calibs[idx]
            corr_pts_idx = torch.nonzero(batch_idxs == gt_idx, as_tuple=False).squeeze(-1)
            center_depth = calib.f_u * pred_h3d[corr_pts_idx] / (
                F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS
            )
            corner_02_depth = calib.f_u * pred_h3d[corr_pts_idx].unsqueeze(-1) / (
                F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS
            )
            corner_13_depth = calib.f_u * pred_h3d[corr_pts_idx].unsqueeze(-1) / (
                F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS
            )

            corner_02_depth = corner_02_depth.mean(dim=1)
            corner_13_depth = corner_13_depth.mean(dim=1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

        pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)
        return pred_depths

    def decode_dim3d(self, cls_id, dim_offset):
        """
        Retrieve object dimensions
        Args:
            cls_id: each object id
            dim_offset: dim offset, with shape of (N, 3)
        """
        cls_id = cls_id.flatten().long()
        cls_dim_mean = self.dim_mean[cls_id, :]

        if self.dim_mode[0] == 'exp':
            dim_offset = dim_offset.exp()
        if self.dim_mode[2]:
            cls_dim_std = self.dim_stds[cls_id, :]
            dimensions = dim_offset * cls_dim_std + cls_dim_mean
        else:
            dimensions = dim_offset * cls_dim_mean

        return dimensions

    def decode_orient(self, local_ori, locations):
        """
        Retrieve object orientation
        Args:
            local_ori (tensor): local orientation in [axis_cls, head_cls, sin, cos] format
            locations (tensor): object location
        Returns: 
            for training we only need roty; for testing we need both alpha and roty
        """
        if self.orient_mode == 'multi-bin':
            pred_bin_cls = local_ori[:, :self.orient_bin_size * 2].view(-1, self.orient_bin_size, 2)
            pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
            orientations = local_ori.new_zeros(local_ori.shape[0])
            for i in range(self.orient_bin_size):
                mask_i = (pred_bin_cls.argmax(dim=1) == i)
                s = self.orient_bin_size * 2 + i * 2
                e = s + 2
                pred_bin_offset = local_ori[mask_i, s:e]
                orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
        else:
            axis_cls = torch.softmax(local_ori[:, :2], dim=1)
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = torch.softmax(local_ori[:, 2:4], dim=1)
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = F.normalize(local_ori[:, 4:])
            orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

        locations = locations.view(-1, 3)
        rays = torch.atan2(locations[:, 0], locations[:, 2])
        alphas = orientations
        rtys = alphas + rays

        large_idx = torch.nonzero(rtys > np.pi, as_tuple=False)
        small_idx = torch.nonzero(rtys < -np.pi, as_tuple=False)

        if len(large_idx) != 0:
            rtys[large_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rtys[small_idx] += 2 * np.pi

        large_idx = torch.nonzero(alphas > np.pi, as_tuple=False)
        small_idx = torch.nonzero(alphas < -np.pi, as_tuple=False)

        if len(large_idx) != 0:
            alphas[large_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            alphas[small_idx] += 2 * np.pi

        return rtys, alphas


if __name__ == '__main__':
    pass
