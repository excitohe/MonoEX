import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from monoex.modeling.utils import fill_fc_weights, get_norm, sigmoid_hm


class MonoFlexPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MonoFlexPredictor, self).__init__()

        self.num_class = len(cfg.DATASETS.CLASS_NAMES)

        self.reg_head_items = cfg.MODEL.HEAD.REG_HEAD_ITEMS
        self.reg_head_chans = cfg.MODEL.HEAD.REG_HEAD_CHANS

        self.output_w = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_h = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO

        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
        self.norm_type = cfg.MODEL.HEAD.NORM
        self.norm_moms = cfg.MODEL.HEAD.NORM_MOMENTUM

        self.inplace_abn = cfg.MODEL.INPLACE_ABN
        self.abn_activation = "leaky_relu"

        # Classification Head
        if self.inplace_abn:
            self.cls_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.norm_moms, activation=self.abn_activation),
                nn.Conv2d(self.head_conv, self.num_class, kernel_size=1, padding=1 // 2, bias=True),
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                get_norm(self.norm_type, self.head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, self.num_class, kernel_size=1, padding=1 // 2, bias=True),
            )
        self.cls_head[-1].bias.data.fill_(-np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        # Regression Head
        self.reg_feats = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for idx, reg_head_key in enumerate(self.reg_head_items):
            if self.inplace_abn:
                reg_pre_feat = nn.Sequential(
                    nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                    InPlaceABN(self.head_conv, momentum=self.norm_moms, activation=self.abn_activation),
                )
            else:
                reg_pre_feat = nn.Sequential(
                    nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                    get_norm(self.norm_type, self.head_conv),
                    nn.ReLU(inplace=True),
                )
            self.reg_feats.append(reg_pre_feat)

            head_chns = self.reg_head_chans[idx]
            head_list = nn.ModuleList()

            for key_idx, key in enumerate(reg_head_key):
                key_chn = head_chns[key_idx]
                reg_out_head = nn.Conv2d(self.head_conv, key_chn, kernel_size=1, padding=1 // 2, bias=True)

                if key.find("uncertainty") >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                    torch.nn.init.xavier_normal_(reg_out_head.weight, gain=0.01)

                if key == 'offset_3d':
                    self.offset_index = [idx, key_idx]

                fill_fc_weights(reg_out_head, value=0)
                head_list.append(reg_out_head)

            self.reg_heads.append(head_list)

        # edge feature
        self.edge_fusion = cfg.MODEL.HEAD.EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU

        if self.edge_fusion:
            tru_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            tru_acti_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            # truncation_heatmap_conv
            self.trunc_hmp_conv = nn.Sequential(
                nn.Conv1d(
                    self.head_conv,
                    self.head_conv,
                    kernel_size=self.edge_fusion_kernel_size,
                    padding=self.edge_fusion_kernel_size // 2,
                    padding_mode='replicate'
                ),
                tru_norm_func(self.head_conv, momentum=self.norm_moms),
                tru_acti_func,
                nn.Conv1d(self.head_conv, self.num_class, kernel_size=1),
            )
            # truncation_offsets_conv
            self.trunc_ofs_conv = nn.Sequential(
                nn.Conv1d(
                    self.head_conv,
                    self.head_conv,
                    kernel_size=self.edge_fusion_kernel_size,
                    padding=self.edge_fusion_kernel_size // 2,
                    padding_mode='replicate'
                ),
                tru_norm_func(self.head_conv, momentum=self.norm_moms),
                tru_acti_func,
                nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

    def forward(self, features, targets):
        batch_size = features.shape[0]

        # output classification
        cls_pre = self.cls_head[:-1](features)
        cls_out = self.cls_head[-1](cls_pre)

        # output regression
        regs_out = []

        for i, reg_pre_head in enumerate(self.reg_feats):
            reg_pre = reg_pre_head(features)

            for j, reg_out_head in enumerate(self.reg_heads[i]):
                reg_out = reg_out_head(reg_pre)

                # edge enhance
                if self.edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    edge_inds = torch.stack([t.get_field("edge_ind") for t in targets])  # B x K x 2
                    edge_lens = torch.stack([t.get_field("edge_len") for t in targets])  # B

                    # normalize
                    grid_edge_inds = edge_inds.view(batch_size, -1, 1, 2).float()
                    grid_edge_inds[..., 0] = grid_edge_inds[..., 0] / (self.output_w - 1) * 2 - 1
                    grid_edge_inds[..., 1] = grid_edge_inds[..., 1] / (self.output_h - 1) * 2 - 1

                    # fuse edge feature for both offsets and heatmap
                    fuse_feat = torch.cat((cls_pre, reg_pre), dim=1)
                    edge_feat = F.grid_sample(fuse_feat, grid_edge_inds, align_corners=True).squeeze(-1)

                    edge_cls_feature = edge_feat[:, :self.head_conv, ...]
                    edge_ofs_feature = edge_feat[:, self.head_conv:, ...]
                    edge_cls_output = self.trunc_hmp_conv(edge_cls_feature)
                    edge_ofs_output = self.trunc_ofs_conv(edge_ofs_feature)

                    for k in range(batch_size):
                        edge_ind_k = edge_inds[k, :edge_lens[k]]
                        cls_out[k, :, edge_ind_k[:, 1], edge_ind_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                        reg_out[k, :, edge_ind_k[:, 1], edge_ind_k[:, 0]] += edge_ofs_output[k, :, :edge_lens[k]]

                regs_out.append(reg_out)

        cls_out = sigmoid_hm(cls_out)
        regs_out = torch.cat(regs_out, dim=1)
        return {'cls': cls_out, 'reg': regs_out}
