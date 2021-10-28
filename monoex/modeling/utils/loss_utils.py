import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon


class VanillaFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(VanillaFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, source, target):
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        pos_loss = torch.log(source) * torch.pow(1 - source, self.gamma) * pos_inds
        neg_loss = torch.log(1 - source) * torch.pow(source, self.gamma) * neg_inds
        pos_nums = pos_inds.float().sum()
        pos_loss = pos_loss.sum() * self.alpha
        neg_loss = neg_loss.sum() * (1 - self.alpha)
        loss = 0.0
        if pos_nums == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / torch.clamp(pos_nums, 1)
        return loss.mean()


class PenaltyFocalLoss(nn.Module):

    def __init__(self, alpha=2, gamma=4):
        super(PenaltyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, source, target):
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weight = torch.pow(1 - target, self.gamma)
        pos_loss = torch.log(source) * torch.pow(1 - source, self.alpha) * pos_inds
        neg_loss = torch.log(1 - source) * torch.pow(source, self.alpha) * neg_inds * neg_weight
        pos_nums = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = 0.0
        if pos_nums == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / torch.clamp(pos_nums, 1)
        return loss.mean()


class DepthRefineLoss(nn.Module):

    def __init__(self, bin_num, bin_size):
        super(DepthRefineLoss, self).__init__()
        self.bin_size = bin_size
        self.bin_num = bin_num
        bin_offset = torch.arange(-bin_num / 2, bin_num / 2)
        self.bin_offset = bin_offset.ceil().view(1, -1) * bin_size

    def forward(self, source, target, refine):
        refine_cls = refine[:, :self.bin_num]
        refine_reg = refine[:, self.bin_num]
        source_with_bin = source.view(-1, 1) + self.bin_offset
        diff_with_bin = source_with_bin - target[:, None]
        target_bin_idx = diff_with_bin.abs().argmin(dim=1)

        # clamp depth diff
        depth_offset = torch.gather(diff_with_bin, 1,
                                    target_bin_idx[:, None]).clamp(min=-self.bin_size / 2,
                                                                   max=self.bin_size / 2).squeeze()
        refine_cls_loss = F.cross_entropy(input=refine_cls, target=target_bin_idx, reduction='none')
        refine_reg_loss = F.smooth_l1_loss(input=refine_reg, target=depth_offset, reduction='none')
        return refine_cls_loss + refine_reg_loss * 10


class BerHuLoss(nn.Module):

    def __init__(self):
        super(BerHuLoss, self).__init__()
        self.c = 0.2

    def forward(self, source, target, weight=None):
        diff = (source - target).abs()
        c = torch.clamp(diff.max() * self.c, min=1e-4)
        # large than c: l2 loss
        # small than c: l1 loss
        large_idx = (diff > c).nonzero()
        small_idx = (diff <= c).nonzero()
        loss = diff[small_idx].sum() + ((diff[large_idx]**2) / c + c).sum() / 2
        if weight is not None:
            loss = loss * weight
        return loss


class InverseSigmoidLoss(nn.Module):

    def __init__(self):
        super(InverseSigmoidLoss, self).__init__()

    def forward(self, source, target, weight=None):
        inverse_sigmoid_sources = 1 / torch.sigmoid(source) - 1
        loss = F.l1_loss(inverse_sigmoid_sources, target, reduction='none')
        if weight is not None:
            loss = loss * weight
        return loss


class LogL1Loss(nn.Module):

    def __init__(self):
        super(LogL1Loss, self).__init__()

    def forward(self, source, target, weight=None):
        loss = F.l1_loss(torch.log(source), torch.log(target), reduction='none')
        if weight is not None:
            loss = loss * weight
        return loss


class InsRelaLoss(nn.Module):

    def __init__(self):
        super(InsRelaLoss, self).__init__()

    def forward(self, source, target, weight=None):
        loss = F.l1_loss(source, torch.log(target), reduction='none')
        if weight is not None:
            loss = loss * weight
        return loss.sum()


class LaplaceLoss(nn.Module):

    def __init__(self):
        super(LaplaceLoss, self).__init__()

    def forward(self, source, target, reduction='none'):
        loss = (1 - source / target).abs()
        return loss


class WingLoss(nn.Module):

    def __init__(self, w=10., eps=2.):
        super(WingLoss, self).__init__()
        self.w = w
        self.eps = eps
        self.c = w - w * np.log(1 + w / eps)

    def forward(self, source, target):
        diff = (source - target).abs()
        log_idxs = (diff < self.w).nonzero()
        l1_idxs = (diff >= self.w).nonzero()

        loss = source.new_zeros(source.shape[0])
        loss[log_idxs] = self.w * torch.log(diff[log_idxs] / self.eps + 1)
        loss[l1_idxs] = diff[l1_idxs] - self.c
        return loss


class MultiBinLoss(nn.Module):

    def __init__(self, num_bin=4):
        super(MultiBinLoss, self).__init__()
        self.num_bin = num_bin

    def forward(self, source, target):
        # bin1_cls, bin1_ofs, bin2_cls, bin2_ofs
        target = target.view(-1, target.shape[-1])
        cls_losses = 0
        reg_losses = 0
        reg_nums = 0
        for i in range(self.num_bin):
            # bin cls loss
            cls_ce_loss = F.cross_entropy(source[:, (i * 2):(i * 2 + 2)], target[:, i].long(), reduction='none')
            # reg loss
            valid_mask_i = (target[:, i] == 1)
            cls_losses += cls_ce_loss.mean()
            if valid_mask_i.sum() > 0:
                s = self.num_bin * 2 + i * 2
                e = s + 2
                source_offset = F.normalize(source[valid_mask_i, s:e])
                reg_loss = F.l1_loss(source_offset[:, 0],
                                     torch.sin(target[valid_mask_i, self.num_bin+i]),
                                     reduction='none') + \
                           F.l1_loss(source_offset[:, 1],
                                     torch.cos(target[valid_mask_i, self.num_bin+i]),
                                     reduction='none')
                reg_losses += reg_loss.sum()
                reg_nums += valid_mask_i.sum()

        return cls_losses / self.num_bin + reg_losses / reg_nums


class UncertaintyRegLoss(nn.Module):

    def __init__(self, reg_loss_func):
        super(UncertaintyRegLoss, self).__init__()
        self.reg_loss_func = reg_loss_func

    def forward(self, source, target, uncerts):
        reg_loss = self.reg_loss_func(source, target)
        reg_loss = reg_loss * torch.exp(-uncerts) + 0.5 * uncerts
        return reg_loss


class UncertaintyLaplaceAleatoricLoss(nn.Module):
    """
    Reference:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    """

    def __init__(self):
        super(UncertaintyLaplaceAleatoricLoss, self).__init__()

    def forward(self, source, target, logvar, reduction='mean'):
        assert reduction in ['mean', 'sum']
        loss = math.sqrt(2) * torch.exp(-0.5 * logvar) * torch.abs(source - target) + 0.5 * logvar
        return loss.mean() if reduction == "mean" else loss.sum()


class UncertaintyGaussianAleatoricLoss(nn.Module):
    """
    Reference:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    """

    def __init__(self):
        super(UncertaintyGaussianAleatoricLoss, self).__init__()

    def forward(self, source, target, logvar, reduction='mean'):
        assert reduction in ['mean', 'sum']
        loss = 0.5 * torch.exp(-logvar) * torch.abs(source - target)**2 + 0.5 * logvar
        return loss.mean() if reduction == "mean" else loss.sum()


class IoULoss(nn.Module):

    def __init__(self, mode="iou"):
        super(IoULoss, self).__init__()
        self.mode = mode

    def forward(self, source, target, weight=None):
        source_left = source[:, 0]
        source_top = source[:, 1]
        source_right = source[:, 2]
        source_bottom = source[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        source_area = (source_left + source_right) * (source_top + source_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(source_left, target_left) + \
                      torch.min(source_right, target_right)
        g_w_intersect = torch.max(source_left, target_left) + \
                        torch.max(source_right, target_right)
        h_intersect = torch.min(source_bottom, target_bottom) + \
                      torch.min(source_top, target_top)
        g_h_intersect = torch.max(source_bottom, target_bottom) + \
                        torch.max(source_top, target_top)

        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + source_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.mode == 'iou':
            losses = -torch.log(ious)
        elif self.mode == 'linear_iou':
            losses = 1 - ious
        elif self.mode == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        return losses, ious


def get_corners_torch(x, y, w, l, yaw):
    """ Temporarily useless
    """
    device = x.device
    bev_corners = torch.zeros((4, 2), dtype=torch.float, device=device)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    return bev_corners


def get_corners(bboxes):
    # bboxes: x, y, w, l, alpha; N x 5
    corners = torch.zeros((bboxes.shape[0], 4, 2), dtype=torch.float, device=bboxes.device)
    x, y, w, l = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # compute cos and sin
    cos_alpha = torch.cos(bboxes[:, -1])
    sin_alpha = torch.sin(bboxes[:, -1])
    # front left
    corners[:, 0, 0] = x - w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 0, 1] = y - w / 2 * sin_alpha + l / 2 * cos_alpha
    # rear left
    corners[:, 1, 0] = x - w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 1, 1] = y - w / 2 * sin_alpha - l / 2 * cos_alpha
    # rear right
    corners[:, 2, 0] = x + w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 2, 1] = y + w / 2 * sin_alpha - l / 2 * cos_alpha
    # front right
    corners[:, 3, 0] = x + w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 3, 1] = y + w / 2 * sin_alpha + l / 2 * cos_alpha
    return corners


def get_iou_3d(source_corners, target_corners):
    """
    Args:
        source_corners (tensor): with shape of [N, 8, 3]
        target_corners (tensor): with shape of [N, 8, 3]
    """
    A, B = source_corners, target_corners
    N = A.shape[0]

    # init output
    iou3d = source_corners.new(N).zero_().float()

    # for height overlap, since y face down, use the negative y
    min_h_a = -A[:, 0:4, 1].sum(dim=1) / 4.0
    max_h_a = -A[:, 4:8, 1].sum(dim=1) / 4.0
    min_h_b = -B[:, 0:4, 1].sum(dim=1) / 4.0
    max_h_b = -B[:, 4:8, 1].sum(dim=1) / 4.0

    # overlap in height
    h_max_of_min = torch.max(min_h_a, min_h_b)
    h_min_of_max = torch.min(max_h_a, max_h_b)
    h_overlap = torch.max(h_min_of_max.new_zeros(h_min_of_max.shape), h_min_of_max - h_max_of_min)

    # x-z plane overlap
    for i in range(N):
        bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]]), Polygon(B[i, 0:4, [0, 2]])
        if bottom_a.is_valid and bottom_b.is_valid:
            # check is valid, A valid Polygon may not possess any
            # overlapping exterior or interior rings.
            bottom_overlap = bottom_a.intersection(bottom_b).area
        else:
            bottom_overlap = 0

        overlap3d = bottom_overlap * h_overlap[i]
        union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + \
                  bottom_b.area * (max_h_b[i] - min_h_b[i]) - overlap3d
        iou3d[i] = overlap3d / union3d

    return iou3d
