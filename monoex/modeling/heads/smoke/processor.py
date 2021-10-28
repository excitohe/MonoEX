import torch
import torch.nn as nn
from monoex.modeling.utils import nms_hm, select_point_of_interest, select_topk
from monoex.modeling.utils.coders import SMOKECoder


class SMOKEProcessor(nn.Module):

    def __init__(self, cfg):
        super(SMOKEProcessor, self).__init__()

        self.coder = SMOKECoder(cfg)
        self.reg_head_items = cfg.MODEL.HEAD.REG_HEAD_ITEMS

        self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG
        self.pred_2d = cfg.TEST.PRED_2D

    def prepare_targets(self, targets, test):
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        size = torch.stack([torch.tensor(t.size) for t in targets])
        if test:
            return dict(K=K, size=size, trans_mat=trans_mat)
        tgt_clsids = torch.stack([t.get_field("tgt_clsids") for t in targets])
        tgt_dim3ds = torch.stack([t.get_field("tgt_dim3ds") for t in targets])
        tgt_rty3ds = torch.stack([t.get_field("tgt_rty3ds") for t in targets])
        tgt_loc3ds = torch.stack([t.get_field("tgt_loc3ds") for t in targets])
        tgt_center_2ds = torch.stack([t.get_field("tgt_center_2ds") for t in targets])
        tgt_corner_3ds = torch.stack([t.get_field("tgt_corner_3ds") for t in targets])
        tgt_fill_mask = torch.stack([t.get_field("tgt_fill_mask") for t in targets])
        target_vars = dict(
            tgt_clsids=tgt_clsids,
            tgt_dim3ds=tgt_dim3ds,
            tgt_rty3ds=tgt_rty3ds,
            tgt_loc3ds=tgt_loc3ds,
            tgt_center_2ds=tgt_center_2ds,
            tgt_corner_3ds=tgt_corner_3ds,
            tgt_fill_mask=tgt_fill_mask,
            K=K,
            size=size,
            trans_mat=trans_mat,
        )
        return target_vars

    def forward(self, predictions, targets, features=None, test=False, refine_module=None):
        pred_hmp, pred_reg = predictions['cls'], predictions['reg']
        batch_size = pred_hmp.shape[0]

        target_vars = self.prepare_targets(targets, test=test)

        heatmap = nms_hm(pred_hmp)
        vis_sources = {
            'heat_map': pred_hmp.clone(),
        }

        scores, indexes, classes, ys, xs = select_topk(heatmap, K=self.max_detection)

        pred_reg_pois = select_point_of_interest(batch_size, indexes, pred_reg).view(-1, pred_reg.shape[1])

        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        pred_dep_offset = pred_reg_pois[:, 0]
        pred_pts_offset = pred_reg_pois[:, 1:3]
        pred_dim_offset = pred_reg_pois[:, 3:6]
        pred_ori_offset = pred_reg_pois[:, 6:]

        pred_depth = self.coder.decode_depth(pred_dep_offset)
        pred_loc3d = self.coder.decode_loc3d(
            pred_proj_points,
            pred_pts_offset,
            pred_depth,
            target_vars['K'],
            target_vars['trans_mat'],
        )
        pred_dim3d = self.coder.decode_dim3d(classes, pred_dim_offset)
        pred_loc3d[:, 1] += pred_dim3d[:, 1] / 2
        pred_rty3d, pred_alpha = self.coder.decode_orient(pred_ori_offset, pred_loc3d)

        if self.pred_2d:
            pred_box2d = self.coder.encode_box2d(
                target_vars['K'],
                pred_rty3d,
                pred_dim3d,
                pred_loc3d,
                target_vars['size'],
            )
        else:
            pred_box2d = torch.tensor([0, 0, 0, 0])

        scores = scores.view(-1, 1)
        classes = classes.view(-1, 1)
        # change gravity_center_location to bottom_center_location
        pred_alpha = pred_alpha.view(-1, 1)
        pred_rty3d = pred_rty3d.view(-1, 1)
        # change dimension back to h, w, l
        pred_dim3d = pred_dim3d.roll(shifts=-1, dims=1)

        result = torch.cat([classes, pred_alpha, pred_box2d, pred_dim3d, pred_loc3d, pred_rty3d, scores], dim=1)

        keep_idx = result[:, -1] >= self.det_threshold
        result = result[keep_idx]
        eval_utils = {
            'dis_ious': None,
            'vis_scores': scores.clone(),
        }

        return result, eval_utils, vis_sources
