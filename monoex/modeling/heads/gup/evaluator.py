import torch
import torch.nn as nn
import torch.nn.functional as F„
from monoex.modeling.utils.htmp_utils import get_source_tensor, get_target_tensor
from monoex.modeling.utils.loss_utils import *


def compute_heading_loss(sources, target_cls, target_reg, mask):
    device = sources.device
    target_cls = target_cls.view(-1)  # [B, K, 1] -> [B*K]
    target_reg = target_reg.view(-1)  # [B, K, 1] -> [B*K]
    mask = mask.view(-1)  # [B, K] -> [B*K]

    # cls loss
    source_cls = sources[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(source_cls, target_cls, reduction='mean')

    # reg loss
    source_reg = sources[:, 12:24]
    target_reg = target_reg[mask]
    onehot_cls = torch.zeros(target_cls.shape[0], 12).to(device).scatter_(
        dim=1, index=target_cls.view(-1, 1), value=1)
    source_reg = torch.sum(source_reg * onehot_cls, 1)
    reg_loss = F.l1_loss(source_reg, target_reg, reduction='mean')
    return cls_loss + reg_loss


class GUPLoss(nn.Module):
    def __init__(self, cfg, epoch):
        super(GUPLoss, self).__init__()

        self.stats = {}
        self.epoch = epoch
        self.hmp_loss_func = PenaltyFocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
                                              cfg.MODEL.HEAD.LOSS_PENALTY_BETA)
        self.dep_loss_func = UncertaintyLaplaceAleatoricLoss()

    def forward(self, sources, targets, task_uncertainties=None):
        sources['heatmap'] = torch.clamp(sources['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        hmp_2d_loss = self.hmp_loss_func(sources['heatmap'], targets['heatmap'])

        # compute dim_2d loss
        source_dim_2d = get_source_tensor(sources['size_2d'], targets['indices'], targets['mask_2d'])
        target_dim_2d = get_target_tensor(targets['size_2d'], targets['mask_2d'])
        dim_2d_loss = F.l1_loss(source_dim_2d, target_dim_2d, reduction='mean')

        # compute ofs_2d loss
        source_ofs_2d = get_source_tensor(sources['offset_2d'], targets['indices'], targets['mask_2d'])
        target_ofs_2d = get_target_tensor(targets['offset_2d'], targets['mask_2d'])
        ofs_2d_loss = F.l1_loss(source_ofs_2d, target_ofs_2d, reduction='mean')

        # compute dep_3d loss
        source_dep_3d = sources['depth'][sources['train_tag']]
        source_dep_3d, source_logvar = source_dep_3d[:, 0:1], source_dep_3d[:, 1:2]
        target_dep_3d = get_target_tensor(targets['depth'], targets['mask_2d'])
        dep_3d_loss = self.dep_loss_func(source_dep_3d, target_dep_3d, source_logvar)

        # compute ofs_3d loss
        source_ofs_3d = sources['offset_3d'][sources['train_tag']]
        target_ofs_3d = get_target_tensor(targets['offset_3d'], targets['mask_2d'])
        ofs_3d_loss = F.l1_loss(source_ofs_3d, target_ofs_3d, reduction='mean')

        # compute dim_3d loss
        source_dim_3d = sources['size_3d'][sources['train_tag']]
        target_dim_3d = get_target_tensor(targets['size_3d'], targets['mask_2d'])
        dim_3d_loss = F.l1_loss(source_dim_3d[:, 1:], target_dim_3d[:, 1:], reduction='mean') * 2 / 3 + \
            self.dep_loss_func(source_dim_3d[:, 0:1], target_dim_3d[:, 0:1], sources['h3d_log_variance'][sources['train_tag']]) / 3

        head_loss = compute_heading_loss(
            sources['heading'][sources['train_tag']],
            targets['heading_bin'],
            targets['heading_res'],
            targets['mask_2d'],
        )

        self.stats['hmp_2d_loss'] = hmp_2d_loss
        self.stats['ofs_2d_loss'] = ofs_2d_loss
        self.stats['dim_2d_loss'] = dim_2d_loss
        self.stats['dep_3d_loss'] = dep_3d_loss
        self.stats['ofs_3d_loss'] = ofs_3d_loss
        self.stats['dim_3d_loss'] = dim_3d_loss
        self.stats['head_loss'] = head_loss

        loss = dep_3d_loss + ofs_3d_loss + dim_3d_loss + head_loss + \
               dim_2d_loss + ofs_2d_loss + hmp_2d_loss

        return loss


class HierarchicalTaskLearn