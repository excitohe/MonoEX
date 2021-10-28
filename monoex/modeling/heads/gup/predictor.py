import numpy as np
import torch
import torch.nn as nn
import torchvision.ops.roi_align as roi_align
from monoex.modeling.utils import (get_source_tensor, init_weights_xavier, nms_hm, select_topk)


class GUPPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(GUPPredictor, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.num_class = len(cfg.DATASETS.DETECT_CLASSES)

        cfg.MODEL.HEAD.INIT_P = 0.1
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG

        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.offset_2d = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.size_2d = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(in_channels + 2 + self.num_class, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.offset_3d = nn.Sequential(
            nn.Conv2d(in_channels + 2 + self.num_class, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.size_3d = nn.Sequential(
            nn.Conv2d(in_channels + 2 + self.num_class, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.heading = nn.Sequential(
            nn.Conv2d(in_channels + 2 + self.num_class, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.heatmap[-1].bias.data.fill_(-np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.depth.apply(init_weights_xavier)
        self.offset_3d.apply(init_weights_xavier)
        self.size_3d.apply(init_weights_xavier)
        self.heading.apply(init_weights_xavier)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features, targets=None):
        batch_size = features.shape[0]

        ret = {}
        ret['heatmap'] = self.heatmap(features)
        ret['offset_2d'] = self.offset_2d(features)
        ret['size_2d'] = self.size_2d(features)

        # two stage
        if self.training:
            print('Training Mode: True')
            indexes = torch.stack([t.get_field("indices") for t in targets])
            classes = torch.stack([t.get_field("clsids") for t in targets])
            masks = torch.stack([t.get_field("mask_2d") for t in targets])
        else:
            print('Training Mode: False')
            heatmap = nms_hm(ret['heatmap'])
            scores, indexes, classes, ys, xs = select_topk(heatmap, K=self.max_detection)
            masks = torch.ones(indexes.size()).type(torch.uint8).to(self.device)

        K = torch.stack([t.get_field("K") for t in targets])
        coord_range = torch.stack([t.get_field("coord_range") for t in targets])

        ret.update(self.get_roi_feat(features, indexes, masks, ret, K, coord_range, classes))
        return ret

    def get_roi_feat(self, feat, inds, mask, rets, calib, coord_ranges, clsids):
        batch_size, _, feat_h, feat_w = feat.size()

        coord_map = torch.cat(
            [
                torch.arange(feat_w).unsqueeze(0).repeat([feat_h, 1]).unsqueeze(0),
                torch.arange(feat_h).unsqueeze(-1).repeat([1, feat_w]).unsqueeze(0)
            ], 0
        ).unsqueeze(0).repeat([batch_size, 1, 1, 1]).type(torch.float).to(self.device)
        box2d_center = coord_map + rets['offset_2d']
        box2d_maps = torch.cat([box2d_center - rets['size_2d'] / 2, box2d_center + rets['size_2d'] / 2], 1)
        box2d_maps = torch.cat(
            [
                torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1, 1, feat_h, feat_w]).type(
                    torch.float
                ).to(self.device), box2d_maps
            ], 1
        )

        # box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat, box2d_maps, inds, mask, calib, coord_ranges, clsids)
        return res

    def get_roi_feat_by_mask(self, feat, box2d_maps, inds, mask, calib, coord_ranges, clsids):
        batch_size, _, feat_h, feat_w = feat.size()
        num_mask_bin = mask.sum()
        res = {}
        if num_mask_bin != 0:
            # get box2d of each roi region
            mask_box2d = get_source_tensor(box2d_maps, inds, mask)
            # get roi feature
            mask_roi_feat = roi_align(feat, mask_box2d, [7, 7])
            # get coord range of each roi
            mask_coord_range_2d = coord_ranges[mask_box2d[:, 0].long()]
            # map box2d coordinate from featmap size domain to original image size domain
            mask_box2d = torch.cat(
                [
                    mask_box2d[:, 0:1], mask_box2d[:, 1:2] / feat_w *
                    (mask_coord_range_2d[:, 1, 0:1] - mask_coord_range_2d[:, 0, 0:1]) + mask_coord_range_2d[:, 0, 0:1],
                    mask_box2d[:, 2:3] / feat_h * (mask_coord_range_2d[:, 1, 1:2] - mask_coord_range_2d[:, 0, 1:2]) +
                    mask_coord_range_2d[:, 0, 1:2], mask_box2d[:, 3:4] / feat_w *
                    (mask_coord_range_2d[:, 1, 0:1] - mask_coord_range_2d[:, 0, 0:1]) + mask_coord_range_2d[:, 0, 0:1],
                    mask_box2d[:, 4:5] / feat_h *
                    (mask_coord_range_2d[:, 1, 1:2] - mask_coord_range_2d[:, 0, 1:2]) + mask_coord_range_2d[:, 0, 1:2]
                ], 1
            )
            roi_calib = calib[mask_box2d[:, 0].long()]
            # project the coord in the normal image to the camera coord by calib
            coord_in_cam = torch.cat(
                [
                    self.project_to_rect(
                        roi_calib, torch.cat([mask_box2d[:, 1:3],
                                              torch.ones([num_mask_bin, 1]).to(self.device)], -1)
                    )[:, :2],
                    self.project_to_rect(
                        roi_calib, torch.cat([mask_box2d[:, 3:5],
                                              torch.ones([num_mask_bin, 1]).to(self.device)], -1)
                    )[:, :2]
                ], -1
            )
            coord_in_cam = torch.cat([mask_box2d[:, 0:1], coord_in_cam], -1)
            # generate coord_maps
            coord_maps = torch.cat(
                [
                    torch.cat(
                        [
                            coord_in_cam[:, 1:2] + i * (coord_in_cam[:, 3:4] - coord_in_cam[:, 1:2]) / 6
                            for i in range(7)
                        ], -1
                    ).unsqueeze(1).repeat([1, 7, 1]).unsqueeze(1),
                    torch.cat(
                        [
                            coord_in_cam[:, 2:3] + i * (coord_in_cam[:, 4:5] - coord_in_cam[:, 2:3]) / 6
                            for i in range(7)
                        ], -1
                    ).unsqueeze(2).repeat([1, 1, 7]).unsqueeze(1)
                ], 1
            )

            # concatenate coord_maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_mask_bin, self.num_class).to(self.device)
            cls_hots[torch.arange(num_mask_bin).to(self.device), clsids[mask].long()] = 1.0

            mask_roi_feat = torch.cat(
                [mask_roi_feat, coord_maps,
                 cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, 7, 7])], 1
            )

            #compute heights of projected objects
            box2d_h = torch.clamp(mask_box2d[:, 4] - mask_box2d[:, 2], min=1.0)
            #compute real 3d height
            size3d_offset = self.size_3d(mask_roi_feat)[:, :, 0, 0]
            h3d_log_std = size3d_offset[:, 3:4]
            size3d_offset = size3d_offset[:, :3]

            size_3d = (self.mean_size[clsids[mask].long()] + size3d_offset)
            depth_geo = size_3d[:, 0] / box2d_h.squeeze() * roi_calib[:, 0, 0]

            depth_net_out = self.depth(mask_roi_feat)[:, :, 0, 0]
            depth_geo_log_std = (h3d_log_std.squeeze() + 2 * (roi_calib[:, 0, 0].log() - box2d_h.log())).unsqueeze(-1)
            depth_net_log_std = torch.logsumexp(
                torch.cat([depth_net_out[:, 1:2], depth_geo_log_std], -1), -1, keepdim=True
            )

            depth_net_out = torch.cat(
                [(1. / (depth_net_out[:, 0:1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1), depth_net_log_std], -1
            )

            res['train_tag'] = torch.ones(num_mask_bin).type(torch.bool).to(self.device)
            res['heading'] = self.heading(mask_roi_feat)[:, :, 0, 0]
            res['depth'] = depth_net_out
            res['offset_3d'] = self.offset_3d(mask_roi_feat)[:, :, 0, 0]
            res['size_3d'] = size3d_offset
            res['h3d_log_variance'] = h3d_log_std
        else:
            res['depth'] = torch.zeros([1, 2]).to(self.device)
            res['offset_3d'] = torch.zeros([1, 2]).to(self.device)
            res['size_3d'] = torch.zeros([1, 3]).to(self.device)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(self.device)
            res['heading'] = torch.zeros([1, 24]).to(self.device)
            res['h3d_log_variance'] = torch.zeros([1, 1]).to(self.device)
        return res

    def project_to_rect(self, calib, point_img):
        c_u = calib[:, 0, 2]
        c_v = calib[:, 1, 2]
        f_u = calib[:, 0, 0]
        f_v = calib[:, 1, 1]
        b_x = calib[:, 0, 3] / (-f_u)  # relative
        b_y = calib[:, 1, 3] / (-f_v)
        x = (point_img[:, 0] - c_u) * point_img[:, 2] / f_u + b_x
        y = (point_img[:, 1] - c_v) * point_img[:, 2] / f_v + b_y
        z = point_img[:, 2]
        object_center = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)
        return object_center
