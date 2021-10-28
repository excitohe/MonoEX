import torch
import torch.nn.functional as F
from monoex.modeling.utils import PenaltyFocalLoss, select_point_of_interest
from monoex.modeling.utils.coders import SMOKECoder


class SMOKEEvaluator(object):

    def __init__(self, cfg):
        super(SMOKEEvaluator, self).__init__()

        self.coder = SMOKECoder(cfg)
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.cls_loss_func = PenaltyFocalLoss(
            cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
            cfg.MODEL.HEAD.LOSS_PENALTY_BETA,
        )

        self.loss_types = cfg.MODEL.HEAD.LOSS_TYPES
        self.loss_names = cfg.MODEL.HEAD.LOSS_NAMES
        self.loss_init_weights = cfg.MODEL.HEAD.LOSS_INIT_WEIGHT

        if self.loss_types[1] == 'L1':
            self.reg_loss_func = F.l1_loss
        else:
            self.reg_loss_func = F.smooth_l1_loss

        self.loss_weights = {}
        for k, w in zip(self.loss_names, self.loss_init_weights):
            self.loss_weights[k] = w

    def prepare_targets(self, targets):
        tgt_htmaps = torch.stack([t.get_field("tgt_htmaps") for t in targets])
        tgt_clsids = torch.stack([t.get_field("tgt_clsids") for t in targets])
        tgt_dim3ds = torch.stack([t.get_field("tgt_dim3ds") for t in targets])
        tgt_loc3ds = torch.stack([t.get_field("tgt_loc3ds") for t in targets])
        tgt_rty3ds = torch.stack([t.get_field("tgt_rty3ds") for t in targets])
        tgt_corner_3ds = torch.stack([t.get_field("tgt_corner_3ds") for t in targets])
        tgt_center_2ds = torch.stack([t.get_field("tgt_center_2ds") for t in targets])
        # utils
        K = torch.stack([t.get_field("K") for t in targets])
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        tgt_fill_mask = torch.stack([t.get_field("tgt_fill_mask") for t in targets])
        tgt_flip_mask = torch.stack([t.get_field("tgt_flip_mask") for t in targets])

        return_dict = dict(
            tgt_htmaps=tgt_htmaps,
            tgt_clsids=tgt_clsids,
            tgt_dim3ds=tgt_dim3ds,
            tgt_loc3ds=tgt_loc3ds,
            tgt_rty3ds=tgt_rty3ds,
            tgt_corner_3ds=tgt_corner_3ds,
            tgt_center_2ds=tgt_center_2ds,
            K=K,
            trans_mat=trans_mat,
            tgt_fill_mask=tgt_fill_mask,
            tgt_flip_mask=tgt_flip_mask,
        )
        return return_dict

    def prepare_sources(self, targets_vars, pred_regs):
        batch, channel = pred_regs.shape[:2]

        # STEP.1: get represent point
        target_centers = targets_vars["tgt_center_2ds"]

        # STEP.2: select point of interest
        pred_reg_pois = select_point_of_interest(batch, target_centers, pred_regs).view(-1, channel)

        # STEP.3: get reg_2d/dim_ofs_3d/ofs_3d/ori_3d from poi
        pred_dep_offset = pred_reg_pois[:, 0]
        pred_pts_offset = pred_reg_pois[:, 1:3]
        pred_dim_offset = pred_reg_pois[:, 3:6]
        pred_ori_offset = pred_reg_pois[:, 6:]

        pred_dep = self.coder.decode_depth(pred_dep_offset)

        pred_loc = self.coder.decode_loc3d(
            target_centers, pred_pts_offset, pred_dep, targets_vars['K'], targets_vars['trans_mat']
        )

        pred_dim = self.coder.decode_dim3d(targets_vars['tgt_clsids'], pred_dim_offset)

        # change gravity_center_location to bottom_center_location
        pred_loc[:, 1] += pred_dim[:, 1] / 2

        pred_rty = self.coder.decode_orient(pred_ori_offset, targets_vars["tgt_loc3ds"], targets_vars["tgt_flip_mask"])

        pred_box3d_rty = self.coder.encode_box3d(pred_rty, targets_vars["tgt_dim3ds"], targets_vars["tgt_loc3ds"])
        pred_box3d_dim = self.coder.encode_box3d(targets_vars["tgt_rty3ds"], pred_dim, targets_vars["tgt_loc3ds"])
        pred_box3d_loc = self.coder.encode_box3d(targets_vars["tgt_rty3ds"], targets_vars["tgt_dim3ds"], pred_loc)

        return_dict = dict(
            ori=pred_box3d_rty,
            dim=pred_box3d_dim,
            loc=pred_box3d_loc,
        )
        return return_dict

    def __call__(self, predictions, targets):
        pred_hmp = predictions['cls']
        pred_reg = predictions['reg']

        targets_vars = self.prepare_targets(targets)
        sources_vars = self.prepare_sources(targets_vars, pred_reg)

        hmp_loss = self.cls_loss_func(pred_hmp, targets_vars["tgt_htmaps"])
        hmp_loss = hmp_loss * self.loss_weights['hmp_loss']

        tgt_corner_3ds = targets_vars['tgt_corner_3ds']
        tgt_corner_3ds = tgt_corner_3ds.view(-1, tgt_corner_3ds.shape[2], tgt_corner_3ds.shape[3])

        tgt_fill_mask = targets_vars['tgt_fill_mask'].flatten().view(-1, 1, 1)
        tgt_fill_mask = tgt_fill_mask.expand_as(tgt_corner_3ds)

        reg_loss_ori = self.reg_loss_func(
            sources_vars['ori'] * tgt_fill_mask, tgt_corner_3ds * tgt_fill_mask, reduction='sum'
        )
        reg_loss_ori = reg_loss_ori / (self.loss_weights['ori_loss'] * self.max_objs)

        reg_loss_dim = self.reg_loss_func(
            sources_vars['dim'] * tgt_fill_mask, tgt_corner_3ds * tgt_fill_mask, reduction='sum'
        )
        reg_loss_dim = reg_loss_dim / (self.loss_weights['dim_loss'] * self.max_objs)

        reg_loss_loc = self.reg_loss_func(
            sources_vars['loc'] * tgt_fill_mask, tgt_corner_3ds * tgt_fill_mask, reduction='sum'
        )
        reg_loss_loc = reg_loss_loc / (self.loss_weights['loc_loss'] * self.max_objs)

        loss_dict = {
            'hmp_loss': hmp_loss,
            'reg_loss': reg_loss_ori + reg_loss_dim + reg_loss_loc,
        }
        log_loss_dict = {}

        # copy loss_dict to log_loss_dict
        for key, value in loss_dict.items():
            if key not in log_loss_dict:
                log_loss_dict[key] = value.item()

        return loss_dict, log_loss_dict
