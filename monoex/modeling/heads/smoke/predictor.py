import torch
import torch.nn as nn
import torch.nn.functional as F
from monoex.modeling.utils import (MakeKeyValuePair, fill_fc_weights, get_norm, sigmoid_hm)


class SMOKEPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(SMOKEPredictor, self).__init__()

        self.num_class = len(cfg.DATASETS.DETECT_CLASSES)

        self.kvper = MakeKeyValuePair(keys=cfg.MODEL.HEAD.REG_HEAD_ITEMS, chns=cfg.MODEL.HEAD.REG_HEAD_CHANS)
        self.sum_chans = sum(self.kvper.chns)

        self.chn_dim_3d = self.kvper('dim_3d')
        self.chn_ori_3d = self.kvper('ori_3d')
        print(f'SMOKE:Predictor:chn_dim_3d: {self.chn_dim_3d}')
        print(f'SMOKE:Predictor:chn_ori_3d: {self.chn_ori_3d}')

        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
        self.norm_type = cfg.MODEL.HEAD.NORM  # GN

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            get_norm(self.norm_type, self.head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_class, kernel_size=1, padding=1 // 2, bias=True),
        )
        self.cls_head[-1].bias.data.fill_(-2.19)

        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            get_norm(self.norm_type, self.head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.sum_chans, kernel_size=1, padding=1 // 2, bias=True),
        )
        fill_fc_weights(self.reg_head)

    def forward(self, features, targets=None):
        cls_out = self.cls_head(features)
        reg_out = self.reg_head(features)

        cls_out = sigmoid_hm(cls_out)
        offset_dim = reg_out[:, self.chn_dim_3d, ...].clone()
        reg_out[:, self.chn_dim_3d, ...] = torch.sigmoid(offset_dim) - 0.5
        vector_ori = reg_out[:, self.chn_ori_3d, ...].clone()
        reg_out[:, self.chn_ori_3d, ...] = F.normalize(vector_ori)

        return {'cls': cls_out, 'reg': reg_out}
