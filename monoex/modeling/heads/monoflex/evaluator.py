import torch
import torch.nn.functional as F

from monoex.modeling.utils.coders import MonoFlexCoder
from monoex.modeling.utils import MakeKeyValuePair, select_point_of_interest
from monoex.modeling.utils.loss_utils import *


class MonoFlexEvaluator(object):

    def __init__(self, cfg):
        super(MonoFlexEvaluator, self).__init__()

        self.coder = MonoFlexCoder(cfg)
        self.kvper = MakeKeyValuePair(
            keys=cfg.MODEL.HEAD.REG_HEAD_ITEMS,
            chns=cfg.MODEL.HEAD.REG_HEAD_CHANS,
        )

        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE

        self.dim_weight = torch.as_tensor(cfg.MODEL.HEAD.DIM_WEIGHT).view(1, 3)
        self.uncert_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

        self.loss_types = cfg.MODEL.HEAD.LOSS_TYPES
        self.loss_names = cfg.MODEL.HEAD.LOSS_NAMES
        self.loss_init_weights = cfg.MODEL.HEAD.LOSS_INIT_WEIGHT

        # cls loss function:
        self.cls_loss_func = PenaltyFocalLoss(
            cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
            cfg.MODEL.HEAD.LOSS_PENALTY_BETA,
        )

        # iou loss function:
        self.iou_loss_func = IoULoss(mode=self.loss_types[2])

        # depth loss function:
        if self.loss_types[3] == 'berhu':
            self.dep_loss_func = BerHuLoss()
        elif self.loss_types[3] == 'inv_sig':
            self.dep_loss_func = InverseSigmoidLoss()
        elif self.loss_types[3] == 'log':
            self.dep_loss_func = LogL1Loss()
        elif self.loss_types[3] == 'L1':
            self.dep_loss_func = F.l1_loss
        else:
            raise ValueError(f"Unsupport depth loss type {self.loss_types[3]}")

        # regression loss function:
        if self.loss_types[1] == 'L1':
            self.reg_loss_func = F.l1_loss
        else:
            self.reg_loss_func = F.smooth_l1_loss

        # keypoint loss function:
        self.kpt_loss_func = F.l1_loss

        # orientation loss function:
        self.orient_mode = cfg.INPUT.ORIENT_MODE
        self.ori_loss_func = MultiBinLoss(num_bin=cfg.INPUT.ORIENT_BIN_SIZE)

        # truncation offset loss function:
        self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

        self.loss_weights = {}
        for k, w in zip(self.loss_names, self.loss_init_weights):
            self.loss_weights[k] = w

        self.enable_corner_loss = 'corner_loss' in self.loss_names
        self.enable_direct_depth_loss = 'dep_loss' in self.loss_names
        self.enable_keypts_depth_loss = 'kpt_dep_loss' in self.loss_names
        self.enable_weight_depth_loss = 'weight_depth_loss' in self.loss_names
        self.enable_trunc_offset_loss = 'trunc_offset_loss' in self.loss_names

        self.enable_direct_depth = 'depth' in self.kvper.keys
        self.enable_uncert_depth = 'depth_uncertainty' in self.kvper.keys
        self.enable_keypts_corner = 'corner_offset' in self.kvper.keys
        self.enable_uncert_corner = 'corner_uncertainty' in self.kvper.keys

        self.modify_invalid_keypts_depth = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

        self.uncert_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT  # 1.0
        self.corner_loss_type = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
        self.eps = 1e-6

    def prepare_targets(self, targets):
        tgt_hmp2ds = torch.stack([t.get_field("tgt_hmp2ds") for t in targets])
        tgt_clsids = torch.stack([t.get_field("tgt_clsids") for t in targets])
        tgt_loc3ds = torch.stack([t.get_field("tgt_loc3ds") for t in targets])
        tgt_dim3ds = torch.stack([t.get_field("tgt_dim3ds") for t in targets])
        tgt_rty3ds = torch.stack([t.get_field("tgt_rty3ds") for t in targets])
        tgt_alphas = torch.stack([t.get_field("tgt_alphas") for t in targets])
        tgt_box2ds = torch.stack([t.get_field("tgt_box2ds") for t in targets])
        tgt_kpt2ds = torch.stack([t.get_field("tgt_kpt2ds") for t in targets])
        tgt_offset_3ds = torch.stack([t.get_field("tgt_offset_3ds") for t in targets])
        tgt_center_2ds = torch.stack([t.get_field("tgt_center_2ds") for t in targets])
        tgt_orient_3ds = torch.stack([t.get_field("tgt_orient_3ds") for t in targets])
        tgt_weight_3ds = torch.stack([t.get_field("tgt_weight_3ds") for t in targets])
        # utils
        calib = [t.get_field("calib") for t in targets]
        pad_size = torch.stack([t.get_field("pad_size") for t in targets])
        tgt_kpt_depth_mask = torch.stack([t.get_field("tgt_kpt_depth_mask") for t in targets])
        tgt_fill_mask = torch.stack([t.get_field("tgt_fill_mask") for t in targets])
        tgt_trun_mask = torch.stack([t.get_field("tgt_trun_mask") for t in targets])
        origin_image = torch.stack([t.get_field("origin_image") for t in targets])

        return_dict = dict(
            tgt_hmp2ds=tgt_hmp2ds,
            tgt_clsids=tgt_clsids,
            tgt_loc3ds=tgt_loc3ds,
            tgt_dim3ds=tgt_dim3ds,
            tgt_rty3ds=tgt_rty3ds,
            tgt_alphas=tgt_alphas,
            tgt_box2ds=tgt_box2ds,
            tgt_kpt2ds=tgt_kpt2ds,
            tgt_offset_3ds=tgt_offset_3ds,
            tgt_center_2ds=tgt_center_2ds,
            tgt_orient_3ds=tgt_orient_3ds,
            tgt_weight_3ds=tgt_weight_3ds,
            calib=calib,
            pad_size=pad_size,
            tgt_kpt_depth_mask=tgt_kpt_depth_mask,
            tgt_fill_mask=tgt_fill_mask,
            tgt_trun_mask=tgt_trun_mask,
            origin_image=origin_image,
        )
        return return_dict

    def prepare_sources(self, targets_vars, predictions):
        pred_reg = predictions['reg']
        batch, channel, _, _ = pred_reg.shape

        # STEP.1: get represent point
        target_centers = targets_vars["tgt_center_2ds"]
        target_fill_mask = targets_vars["tgt_fill_mask"]
        target_fill_mask_flat = target_fill_mask.view(-1).bool()

        # STEP.2: get batch_inds for each object for pad_size, calib and so on.
        batch_inds = torch.arange(batch).view(-1, 1).expand_as(target_fill_mask).reshape(-1)
        batch_inds = batch_inds[target_fill_mask_flat].to(target_fill_mask.device)
        target_box_pts_valid = target_centers.view(-1, 2)[target_fill_mask_flat]

        # STEP.3: get target_box_2d in fcos style
        target_box_2d = targets_vars['tgt_box2ds'].view(-1, 4)[target_fill_mask_flat]
        target_box_2d_h = target_box_2d[:, 3] - target_box_2d[:, 1]
        target_box_2d_w = target_box_2d[:, 2] - target_box_2d[:, 0]

        # STEP.4: get target_reg_2d
        target_reg_2d = torch.cat(
            [
                target_box_pts_valid - target_box_2d[:, :2],
                target_box_2d[:, 2:] - target_box_pts_valid,
            ], dim=1
        )

        masked_reg_2d = (target_box_2d_h > 0) & (target_box_2d_w > 0)
        target_reg_2d = target_reg_2d[masked_reg_2d]

        # STEP.5: target_vars for 3d info
        target_cls_id = targets_vars["tgt_clsids"].view(-1)[target_fill_mask_flat]
        target_dim_3d = targets_vars['tgt_dim3ds'].view(-1, 3)[target_fill_mask_flat]
        target_rty_3d = targets_vars['tgt_rty3ds'].view(-1)[target_fill_mask_flat]
        target_dep_3d = targets_vars['tgt_loc3ds'][..., -1].view(-1)[target_fill_mask_flat]
        target_offset_3d = targets_vars["tgt_offset_3ds"].view(-1, 2)[target_fill_mask_flat]
        target_orient_3d = targets_vars['tgt_orient_3ds'].view(-1, targets_vars['tgt_orient_3ds'].shape[-1]
                                                               )[target_fill_mask_flat]
        target_weight_3d = targets_vars["tgt_weight_3ds"].view(-1)[target_fill_mask_flat]
        target_tru_mask = targets_vars['tgt_trun_mask'].view(-1)[target_fill_mask_flat]

        # STEP.6: compute loc3d
        target_loc_3d = self.coder.decode_loc3d(
            target_box_pts_valid, target_offset_3d, target_dep_3d, targets_vars['calib'], targets_vars['pad_size'],
            batch_inds
        )

        # STEP.7: concat loc3d/dim3d/rty3d as box3d
        target_box_3d = torch.cat((target_loc_3d, target_dim_3d, target_rty_3d[:, None]), dim=1)

        # STEP.8: encode_box3d
        target_corner_3d = self.coder.encode_box3d(target_rty_3d, target_dim_3d, target_loc_3d)

        # STEP.9: select point of interest
        pred_reg_pois = select_point_of_interest(batch, target_centers, pred_reg).view(-1,
                                                                                       channel)[target_fill_mask_flat]

        # STEP.10: get reg_2d/dim_ofs_3d/ofs_3d/ori_3d from poi
        pred_reg_2d = F.relu(pred_reg_pois[masked_reg_2d, self.kvper('dim_2d')])
        pred_dim_offset_3d = pred_reg_pois[:, self.kvper('dim_3d')]
        pred_offset_3d = pred_reg_pois[:, self.kvper('offset_3d')]
        pred_orient_3d = torch.cat(
            [
                pred_reg_pois[:, self.kvper('orient_cls')],
                pred_reg_pois[:, self.kvper('orient_ofs')],
            ], dim=1
        )

        # STEP.11: compute dim3d from residual to actual
        pred_dim_3d = self.coder.decode_dim3d(target_cls_id, pred_dim_offset_3d)

        # STEP.12: reformat targets/sources/regnums/weights
        targets = {
            'reg_2d': target_reg_2d,
            'dim_3d': target_dim_3d,
            'rty_3d': target_rty_3d,
            'cat_3d': target_box_3d,
            'depth_3d': target_dep_3d,
            'offset_3d': target_offset_3d,
            'orient_3d': target_orient_3d,
            'corner_3d': target_corner_3d,
            'h_2d': target_box_2d_h,
            'w_2d': target_box_2d_w,
            'tru_mask_3d': target_tru_mask,
        }
        sources = {
            'reg_2d': pred_reg_2d,
            'dim_3d': pred_dim_3d,
            'offset_3d': pred_offset_3d,
            'orient_3d': pred_orient_3d,
        }
        regnums = {
            'reg_2d': masked_reg_2d.sum(),
            'reg_3d': target_fill_mask_flat.sum(),
        }
        weights = {
            'target_weights': target_weight_3d,
        }

        # STEP.13: compute direct depth
        if self.enable_direct_depth:
            pred_depth_offset_3d = pred_reg_pois[:, self.kvper('depth')].squeeze(-1)
            pred_depth_direct_3d = self.coder.decode_depth(pred_depth_offset_3d)
            sources['depth_3d'] = pred_depth_direct_3d

        # STEP.14: compute uncert depth
        if self.enable_uncert_depth:
            sources['depth_uncertainty'] = \
                pred_reg_pois[:, self.kvper('depth_uncertainty')].squeeze(-1)
            if self.uncert_range is not None:
                sources['depth_uncertainty'] = torch.clamp(
                    sources['depth_uncertainty'], min=self.uncert_range[0], max=self.uncert_range[1]
                )

        # STEP.15: compute keypoint corner
        if self.enable_keypts_corner:
            # reformat targets for keypoint
            target_corner_kpt = targets_vars["tgt_kpt2ds"].view(target_fill_mask_flat.shape[0], -1,
                                                                 3)[target_fill_mask_flat]
            targets['kpt_2d'] = target_corner_kpt[..., :2]
            targets['kpt_2d_mask'] = target_corner_kpt[..., -1]
            regnums['kpt_2d'] = targets['kpt_2d_mask'].sum()
            targets['kpt_depth_mask'] = targets_vars["tgt_kpt_depth_mask"].view(-1, 3)[target_fill_mask_flat]

            # reformat compute for keypoint
            pred_kpt_2d = pred_reg_pois[:, self.kvper('corner_offset')]
            pred_kpt_2d = pred_kpt_2d.view(target_fill_mask_flat.sum(), -1, 2)
            pred_kpt_depth_2d = self.coder.decode_depth_from_kpts_batch(
                pred_kpt_2d, pred_dim_3d, targets_vars['calib'], batch_inds
            )
            sources['kpt_2d'] = pred_kpt_2d
            sources['kpt_depth'] = pred_kpt_depth_2d

        # STEP.16: compute the uncert of the solved depths from groups of keypoints
        if self.enable_uncert_corner:
            sources['corner_offset_uncertainty'] = \
                pred_reg_pois[:, self.kvper('corner_uncertainty')]
            if self.uncert_range is not None:
                sources['corner_offset_uncertainty'] = torch.clamp(
                    sources['corner_offset_uncertainty'], min=self.uncert_range[0], max=self.uncert_range[1]
                )

        # STEP.17: compute the corners of the predicted box3d
        if self.corner_loss_type == 'direct':
            pred_corner_depth_3d = pred_depth_direct_3d
        elif self.corner_loss_type == 'keypoint_mean':
            pred_corner_depth_3d = sources['kpt_depth'].mean(dim=1)
        else:
            assert self.corner_loss_type in ['soft_combine', 'hard_combine']
            pred_combine_uncert = torch.cat(
                [
                    sources['depth_uncertainty'].unsqueeze(-1),
                    sources['corner_offset_uncertainty'],
                ], dim=1
            ).exp()
            pred_combine_depth = torch.cat([
                pred_depth_direct_3d.unsqueeze(-1),
                sources['kpt_depth'],
            ], dim=1)

            if self.corner_loss_type == 'soft_combine':
                pred_uncert_weight = 1 / pred_combine_uncert
                pred_uncert_weight = pred_uncert_weight / pred_uncert_weight.sum(dim=1, keepdim=True)
                pred_corner_depth_3d = torch.sum(pred_combine_depth * pred_uncert_weight, dim=1)
                sources['weighted_depths'] = pred_corner_depth_3d
            elif self.corner_loss_type == 'hard_combine':
                pred_corner_depth_3d = pred_combine_depth[torch.arange(pred_combine_depth.shape[0]),
                                                          pred_combine_uncert.argmin(dim=1)]

        # STEP.18: compute loc3d
        pred_loc_3d = self.coder.decode_loc3d(
            target_box_pts_valid, pred_offset_3d, pred_corner_depth_3d, targets_vars['calib'], targets_vars['pad_size'],
            batch_inds
        )

        # STEP.19: decode rty3ds and alphas
        pred_rty_3d, _ = self.coder.decode_orient(pred_orient_3d, pred_loc_3d)
        pred_box_3d = torch.cat((pred_loc_3d, pred_dim_3d, pred_rty_3d[:, None]), dim=1)
        pred_cor_3d = self.coder.encode_box3d(pred_rty_3d, pred_dim_3d, pred_loc_3d)

        sources.update({
            'corner_3d': pred_cor_3d,
            'rty_3d': pred_rty_3d,
            'cat_3d': pred_box_3d,
        })
        return targets, sources, regnums, weights

    def __call__(self, predictions, targets):
        pred_heatmap = predictions['cls']
        targets_vars = self.prepare_targets(targets)

        _targets, _sources, regnums, weights = self.prepare_sources(targets_vars, predictions)

        if self.heatmap_type == 'centernet':
            # LOSS: hmp_2d
            hmp_loss = self.cls_loss_func(pred_heatmap, targets_vars["tgt_hmp2ds"])
            hmp_loss = self.loss_weights['hmp_loss'] * hmp_loss
        else:
            raise ValueError(f"Unsupport heatmap loss {self.heatmap_type}")

        # synthesize normal factors
        num_reg_2d = regnums['reg_2d']
        num_reg_3d = regnums['reg_3d']
        tgt_trun_mask = _targets['tru_mask_3d'].bool()

        if num_reg_2d > 0:
            # LOSS: iou_2d
            reg_2d_loss, iou_2d = self.iou_loss_func(_sources['reg_2d'], _targets['reg_2d'])
            reg_2d_loss = self.loss_weights['box_loss'] * reg_2d_loss.mean()
            iou_2d = iou_2d.mean()
            depth_MAE = (_sources['depth_3d'] - _targets['depth_3d']).abs() / _targets['depth_3d']

        if num_reg_3d > 0:
            # LOSS: dep_3d
            if self.enable_direct_depth_loss:
                dep_3d_loss = self.dep_loss_func(_sources['depth_3d'], _targets['depth_3d'], reduction='none')
                dep_3d_loss = self.loss_weights['dep_loss'] * dep_3d_loss
                real_dep_3d_loss = dep_3d_loss.detach().mean()
                if self.enable_uncert_depth:
                    dep_3d_loss = dep_3d_loss * torch.exp(- _sources['depth_uncertainty']) + \
                                  _sources['depth_uncertainty'] * self.loss_weights['dep_loss']
                dep_3d_loss = dep_3d_loss.mean()

            # LOSS: ofs_3d
            offset_3d_loss = self.reg_loss_func(_sources['offset_3d'], _targets['offset_3d'],
                                                reduction='none').sum(dim=1)

            # separate consider inside and outside objects
            if self.enable_trunc_offset_loss:
                if self.trunc_offset_loss_type == 'L1':
                    trunc_offset_loss = offset_3d_loss[tgt_trun_mask]
                elif self.trunc_offset_loss_type == 'log':
                    trunc_offset_loss = torch.log(1 + offset_3d_loss[tgt_trun_mask])
                trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum(
                ) / torch.clamp(tgt_trun_mask.sum(), min=1)
                offset_3d_loss = self.loss_weights['offset_loss'] * offset_3d_loss[~tgt_trun_mask].mean()
            else:
                offset_3d_loss = self.loss_weights['offset_loss'] * offset_3d_loss.mean()

            # LOSS: ori_3d
            if self.orient_mode == 'multi-bin':
                ori_3d_loss = self.ori_loss_func(_sources['orient_3d'], _targets['orient_3d'])
                ori_3d_loss = self.loss_weights['ori_loss'] * ori_3d_loss

            # LOSS: dim_3d
            dim_3d_loss = self.reg_loss_func(_sources['dim_3d'], _targets['dim_3d'],
                                             reduction='none') * self.dim_weight.type_as(_sources['dim_3d'])
            dim_3d_loss = self.loss_weights['dim_loss'] * dim_3d_loss.sum(dim=1).mean()

            with torch.no_grad():
                pred_iou_3d = get_iou_3d(_sources['corner_3d'], _targets['corner_3d']).mean()

            # LOSS: cor_3d
            if self.enable_corner_loss:
                corner_3d_loss = self.reg_loss_func(_sources['corner_3d'], _targets['corner_3d'],
                                                    reduction='none').sum(dim=2).mean()
                corner_3d_loss = self.loss_weights['corner_loss'] * corner_3d_loss

            if self.enable_keypts_corner:
                kpt_loss = self.kpt_loss_func(_sources['kpt_2d'], _targets['kpt_2d'], reduction='none').sum(dim=2)
                kpt_loss = self.loss_weights['kpt_loss'] * kpt_loss * _targets['kpt_2d_mask']
                kpt_loss = kpt_loss.sum() / torch.clamp(_targets['kpt_2d_mask'].sum(), min=1)

                if self.enable_keypts_depth_loss:
                    kpt_depth_pred = _sources['kpt_depth']
                    kpt_depth_mask = _targets['kpt_depth_mask'].bool()

                    target_kpt_depth = _targets['depth_3d'].unsqueeze(-1).repeat(1, 3)

                    valid_pred_kpt_depth = kpt_depth_pred[kpt_depth_mask]
                    invalid_pred_kpt_depth = kpt_depth_pred[~kpt_depth_mask].detach()

                    # valid and non-valid
                    valid_kpt_depth_loss = self.reg_loss_func(
                        valid_pred_kpt_depth, target_kpt_depth[kpt_depth_mask], reduction='none'
                    )
                    valid_kpt_depth_loss = self.loss_weights['kpt_dep_loss'] * valid_kpt_depth_loss

                    invalid_kpt_depth_loss = self.reg_loss_func(
                        invalid_pred_kpt_depth, target_kpt_depth[~kpt_depth_mask], reduction='none'
                    )
                    invalid_kpt_depth_loss = self.loss_weights['kpt_dep_loss'] * invalid_kpt_depth_loss

                    # for logging
                    log_valid_kpt_depth_loss = valid_kpt_depth_loss.detach().mean()

                    if self.enable_uncert_corner:
                        # center depth, corner 0246 depth, corner 1357 depth
                        pred_kpt_depth_uncertainty = _sources['corner_offset_uncertainty']
                        valid_uncertainty = pred_kpt_depth_uncertainty[kpt_depth_mask]
                        invalid_uncertainty = pred_kpt_depth_uncertainty[~kpt_depth_mask]
                        valid_kpt_depth_loss = valid_kpt_depth_loss * torch.exp(-valid_uncertainty) + self.loss_weights[
                            'kpt_dep_loss'] * valid_uncertainty
                        invalid_kpt_depth_loss = invalid_kpt_depth_loss * torch.exp(-invalid_uncertainty)

                    # average
                    valid_kpt_depth_loss = valid_kpt_depth_loss.sum() / torch.clamp(kpt_depth_mask.sum(), 1)
                    invalid_kpt_depth_loss = invalid_kpt_depth_loss.sum() / torch.clamp((~kpt_depth_mask).sum(), 1)

                    # the gradients of invalid depths are not back-propagated
                    if self.modify_invalid_keypts_depth:
                        kpt_dep_loss = valid_kpt_depth_loss + invalid_kpt_depth_loss
                    else:
                        kpt_dep_loss = valid_kpt_depth_loss

                # compute the average error for each method of depth estimation
                kpt_MAE = (_sources['kpt_depth'] -
                           _targets['depth_3d'].unsqueeze(-1)).abs() / _targets['depth_3d'].unsqueeze(-1)
                center_MAE = kpt_MAE[:, 0].mean()
                kpt_02_MAE = kpt_MAE[:, 1].mean()
                kpt_13_MAE = kpt_MAE[:, 2].mean()

                if self.enable_uncert_corner:
                    if self.enable_direct_depth and self.enable_uncert_depth:
                        combined_depth = torch.cat((_sources['depth_3d'].unsqueeze(1), _sources['kpt_depth']), dim=1)
                        combined_uncertainty = torch.cat(
                            (_sources['depth_uncertainty'].unsqueeze(1), _sources['corner_offset_uncertainty']), dim=1
                        ).exp()
                        combined_MAE = torch.cat((depth_MAE.unsqueeze(1), kpt_MAE), dim=1)
                    else:
                        combined_depth = _sources['kpt_depth']
                        combined_uncertainty = _sources['corner_offset_uncertainty'].exp()
                        combined_MAE = kpt_MAE

                    # Oracle MAE
                    lower_MAE = torch.min(combined_MAE, dim=1)[0]
                    # the hard ensemble
                    hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
                    # the soft ensemble
                    combined_weights = 1 / combined_uncertainty
                    combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
                    soft_depth = torch.sum(combined_depth * combined_weights, dim=1)
                    soft_MAE = (soft_depth - _targets['depth_3d']).abs() / _targets['depth_3d']
                    # the average ensemble
                    mean_depth = combined_depth.mean(dim=1)
                    mean_MAE = (mean_depth - _targets['depth_3d']).abs() / _targets['depth_3d']

                    # average
                    lower_MAE = lower_MAE.mean()
                    hard_MAE = hard_MAE.mean()
                    soft_MAE = soft_MAE.mean()
                    mean_MAE = mean_MAE.mean()

                    if self.enable_weight_depth_loss:
                        soft_depth_loss = self.reg_loss_func(soft_depth, _targets['depth_3d'], reduction='mean')
                        soft_depth_loss = self.loss_weights['weight_depth_loss'] * soft_depth_loss

            depth_MAE = depth_MAE.mean()

        loss_dict = {
            'hmp_loss': hmp_loss,
            'box_loss': reg_2d_loss,
            'dim_loss': dim_3d_loss,
            'ori_loss': ori_3d_loss,
        }
        log_loss_dict = {
            'iou2d': iou_2d.item(),
            'iou3d': pred_iou_3d.item(),
        }
        MAE_dict = {}

        if self.enable_trunc_offset_loss:
            loss_dict['offset_loss'] = offset_3d_loss
            loss_dict['trunc_offset_loss'] = trunc_offset_loss
        else:
            loss_dict['offset_loss'] = offset_3d_loss

        if self.enable_corner_loss:
            loss_dict['corner_loss'] = corner_3d_loss

        if self.enable_direct_depth:
            loss_dict['dep_loss'] = dep_3d_loss
            log_loss_dict['dep_loss'] = real_dep_3d_loss.item()
            MAE_dict['depth_MAE'] = depth_MAE.item()

        if self.enable_keypts_corner:
            loss_dict['kpt_loss'] = kpt_loss

            MAE_dict.update(
                {
                    'center_MAE': center_MAE.item(),
                    '02_MAE': kpt_02_MAE.item(),
                    '13_MAE': kpt_13_MAE.item(),
                }
            )

            if self.enable_uncert_corner:
                MAE_dict.update(
                    {
                        'lower_MAE': lower_MAE.item(),
                        'hard_MAE': hard_MAE.item(),
                        'soft_MAE': soft_MAE.item(),
                        'mean_MAE': mean_MAE.item(),
                    }
                )

        if self.enable_keypts_depth_loss:
            loss_dict['kpt_dep_loss'] = kpt_dep_loss
            log_loss_dict['kpt_dep_loss'] = log_valid_kpt_depth_loss.item()

        if self.enable_weight_depth_loss:
            loss_dict['weight_depth_loss'] = soft_depth_loss

        # copy loss_dict to log_loss_dict
        for key, value in loss_dict.items():
            if key not in log_loss_dict:
                log_loss_dict[key] = value.item()

        log_loss_dict.update(MAE_dict)
        return loss_dict, log_loss_dict
