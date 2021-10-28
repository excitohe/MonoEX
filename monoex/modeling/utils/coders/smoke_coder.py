import math
import torch


class SMOKECoder(object):

    def __init__(self, cfg):
        super(SMOKECoder, self).__init__()
        device = cfg.MODEL.DEVICE
        self.EPS = 1e-7
        self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
        self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REF).to(device=device)
        self.dim_ref = torch.as_tensor(cfg.MODEL.HEAD.DIM_REF).to(device=device)

    def encode_box2d(self, K, rtys, dims, locs, image_size):
        device = rtys.device
        K = K.to(device=device)
        image_size = image_size.flatten()
        box3d = self.encode_box3d(rtys, dims, locs)
        box3d_image = torch.matmul(K, box3d)
        box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(box3d.shape[0], 1, box3d.shape[2])

        xmins, _ = box3d_image[:, 0, :].min(dim=1)
        xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
        ymins, _ = box3d_image[:, 1, :].min(dim=1)
        ymaxs, _ = box3d_image[:, 1, :].max(dim=1)
        xmins = xmins.clamp(0, image_size[0])
        xmaxs = xmaxs.clamp(0, image_size[0])
        ymins = ymins.clamp(0, image_size[1])
        ymaxs = ymaxs.clamp(0, image_size[1])
        box_from3d = torch.cat(
            [
                xmins.unsqueeze(1),
                ymins.unsqueeze(1),
                xmaxs.unsqueeze(1),
                ymaxs.unsqueeze(1),
            ], dim=1
        )
        return box_from3d

    @staticmethod
    def rad_to_matrix(rtys, N):
        device = rtys.device
        cos, sin = rtys.cos(), rtys.sin()
        tmp = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [-1, 0, 1],
        ]).to(dtype=torch.float32, device=device)
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
        dims = dims.view(-1, 1).repeat(1, 8)
        dims[::3, :4] = 0.5 * dims[::3, :4]
        dims[::3, 4:] = -0.5 * dims[::3, 4:]
        dims[2::3, :4] = 0.5 * dims[2::3, :4]
        dims[2::3, 4:] = -0.5 * dims[2::3, 4:]
        dims[1::3, :4] = 0.
        dims[1::3, 4:] = -dims[1::3, 4:]
        index = torch.tensor([
            [4, 0, 1, 2, 3, 5, 6, 7],
            [4, 5, 0, 1, 6, 7, 2, 3],
            [4, 5, 6, 0, 1, 2, 3, 7],
        ]).repeat(N, 1).to(device=device)
        box_3d_object = torch.gather(dims, 1, index)
        box_3d = torch.matmul(mat, box_3d_object.view(N, 3, -1))
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)
        return box_3d

    def decode_depth(self, offsets):
        if self.depth_mode == 'exp':
            depth = offsets.exp()
        elif self.depth_mode == 'linear':
            depth = offsets * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(offsets) - 1
        else:
            raise ValueError(f"Unsupport depth decode mode {self.depth_mode}")
        return depth

    def decode_loc3d(self, points, offsets, depths, calibs, trans_mats):
        """ Decode location
        Args:
            points (tensor): project center
            offsets (tensor): project offset
            depths (tensor): 
            calibs (tensor): camera intrinsic P2
        Returns: 
            locations (tensor): box3d gravity center
        """
        device = points.device
        calibs = calibs.to(device=device)
        trans_mats = trans_mats.to(device=device)
        num = offsets.shape[0]
        num_batch = calibs.shape[0]
        batch_id = torch.arange(num_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, num // num_batch).flatten()
        trans_mats_inv = trans_mats.inverse()[obj_id]
        calibs_inv = calibs.inverse()[obj_id]

        points = points.view(-1, 2)
        assert points.shape[0] == num
        project_points = points + offsets
        project_points_ext = torch.cat((project_points, torch.ones(num, 1).to(device=device)), dim=1)
        project_points_ext = project_points_ext.unsqueeze(-1)
        project_points_img = torch.matmul(trans_mats_inv, project_points_ext)
        project_points_img = project_points_img * depths.view(num, -1, 1)
        locations = torch.matmul(calibs_inv, project_points_img)
        return locations.squeeze(2)

    def decode_dim3d(self, cls_id, dims_offset):
        cls_id = cls_id.flatten().long()
        dims_select = self.dim_ref[cls_id, :]
        dims = dims_offset.exp() * dims_select
        return dims

    def decode_orient(self, local_ori, locations, flip_mask=None):
        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + self.EPS))
        alphas = torch.atan(local_ori[:, 0] / (local_ori[:, 1] + self.EPS))
        cos_pos_idx = torch.nonzero(local_ori[:, 1] >= 0, as_tuple=False)
        cos_neg_idx = torch.nonzero(local_ori[:, 1] < 0, as_tuple=False)
        alphas[cos_pos_idx] -= math.pi / 2
        alphas[cos_neg_idx] += math.pi / 2
        rtys = alphas + rays

        large_idx = torch.nonzero(rtys > math.pi, as_tuple=False)
        small_idx = torch.nonzero(rtys < -math.pi, as_tuple=False)
        if len(large_idx) != 0:
            rtys[large_idx] -= 2 * math.pi
        if len(small_idx) != 0:
            rtys[small_idx] += 2 * math.pi

        if flip_mask is not None:
            flip_mask_flatten = flip_mask.flatten()
            rtys_flip = flip_mask_flatten.float() * rtys
            rtys_flip_pos_idx = rtys_flip > 0
            rtys_flip_neg_idx = rtys_flip < 0
            rtys_flip[rtys_flip_pos_idx] -= math.pi
            rtys_flip[rtys_flip_neg_idx] += math.pi
            rtys_all = flip_mask_flatten.float() * rtys_flip + \
                       (1 - flip_mask_flatten.float()) * rtys
            return rtys_all
        else:
            return rtys, alphas


if __name__ == '__main__':
    from umi.config import get_cfg
    cfg = get_cfg()
    cfg.MODEL.HEAD.DEPTH_MODE = "linear"
    cfg.MODEL.HEAD.DEPTH_REF = (28.01, 16.32)

    device = cfg.MODEL.DEVICE

    smoke_coder = SMOKECoder(cfg)
    depth_offset = torch.tensor([-1.3977, -0.9933, 0.0000, -0.7053]).to(device=device)
    depth = smoke_coder.decode_depth(depth_offset)
    print('depth:')
    print(depth)

    points = torch.tensor([[4, 75], [200, 59], [0, 0], [97, 54]]).to(device=device)
    offsets = torch.tensor([[0.5722, 0.1508], [0.6010, 0.1145], [0.0000, 0.0000], [0.0365, 0.1977]]).to(device=device)
    calib = torch.tensor([[721.54, 0., 631.44], [0., 721.54, 172.85], [0, 0, 1]]).to(device=device).unsqueeze(0)
    trans_mat = torch.tensor(
        [[2.5765e-01, -0.0000e+00, 2.5765e-01], [-2.2884e-17, 2.5765e-01, -3.0918e-01], [0, 0, 1]]
    ).to(device=device).unsqueeze(0)
    locations = smoke_coder.decode_loc3d(points, offsets, depth, calib, trans_mat)
    # NOTE: this location is box3d_gravity_center

    cls_ids = torch.tensor([[0], [0], [0], [0]]).to(device=device)

    dim_offsets = torch.tensor(
        [[-0.0375, 0.0755, -0.1469], [-0.1309, 0.1054, 0.0179], [0.0000, 0.0000, 0.0000], [-0.0765, 0.0447, -0.1803]]
    ).to(device=device).roll(1, 1)
    dimensions = smoke_coder.decode_dim3d(cls_ids, dim_offsets)
    print('loc3d:')
    print(locations)
    print('dim3d:')
    print(dimensions)

    vector_ori = torch.tensor([[0.4962, 0.8682], [0.3702, -0.9290], [0.0000, 0.0000], [0.2077,
                                                                                       0.9782]]).to(device=device)
    flip_mask = torch.tensor([1, 1, 0, 1]).to(device=device)
    rotys = smoke_coder.decode_orient(vector_ori, locations, flip_mask)
    print('rty3d:')
    print(rotys)

    rotys = torch.tensor([[1.4200], [-1.7600], [0.0000], [1.4400]]).to(device=device)
    box3d = smoke_coder.encode_box3d(rotys, dimensions, locations)
    print('box3d:')
    print(box3d)
