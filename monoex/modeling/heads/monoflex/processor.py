import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
from monoex.engine.visual_api import box_iou
from monoex.modeling.utils import (MakeKeyValuePair, nms_hm, select_point_of_interest, select_topk)
from monoex.modeling.utils.coders import MonoFlexCoder


class MonoFlexProcessor(nn.Module):

    def __init__(self, cfg):
        super(MonoFlexProcessor, self).__init__()

        self.coder = MonoFlexCoder(cfg)
        self.kvper = MakeKeyValuePair(keys=cfg.MODEL.HEAD.REG_HEAD_ITEMS, chns=cfg.MODEL.HEAD.REG_HEAD_CHANS)

        self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG
        self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
        self.eval_depth = cfg.TEST.EVAL_DEPTH

        self.output_w = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_h = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH
        self.pred_2d = cfg.TEST.PRED_2D

        self.enable_direct_depth = 'depth' in self.kvper.keys
        self.enable_uncert_depth = 'depth_uncertainty' in self.kvper.keys
        self.enable_keypts_corner = 'corner_offset' in self.kvper.keys
        self.enable_uncert_corner = 'corner_uncertainty' in self.kvper.keys

        # use uncertainty to guide the confidence
        self.enable_uncert_confidence = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE

    def prepare_targets(self, targets, test):
        calib = [t.get_field("calib") for t in targets]
        pad_size = torch.stack([t.get_field("pad_size") for t in targets])
        img_size = torch.stack([torch.tensor(t.size) for t in targets])
        if test:
            return dict(calib=calib, img_size=img_size, pad_size=pad_size)
        tgt_clsids = torch.stack([t.get_field("tgt_clsids") for t in targets])
        tgt_dim3ds = torch.stack([t.get_field("tgt_dim3ds") for t in targets])
        tgt_rty3ds = torch.stack([t.get_field("tgt_rty3ds") for t in targets])
        tgt_loc3ds = torch.stack([t.get_field("tgt_loc3ds") for t in targets])
        tgt_center_2ds = torch.stack([t.get_field("tgt_center_2ds") for t in targets])
        tgt_offset_3ds = torch.stack([t.get_field("tgt_offset_3ds") for t in targets])
        tgt_fill_mask = torch.stack([t.get_field("tgt_fill_mask") for t in targets])
        target_vars = dict(
            calib=calib,
            img_size=img_size,
            pad_size=pad_size,
            tgt_clsids=tgt_clsids,
            tgt_loc3ds=tgt_loc3ds,
            tgt_dim3ds=tgt_dim3ds,
            tgt_rty3ds=tgt_rty3ds,
            tgt_center_2ds=tgt_center_2ds,
            tgt_offset_3ds=tgt_offset_3ds,
            tgt_fill_mask=tgt_fill_mask,
        )
        return target_vars

    def forward(self, predictions, targets, features=None, test=False, refine_module=None):
        pred_hmp, pred_reg = predictions['cls'], predictions['reg']
        batch_size = pred_hmp.shape[0]

        target_vars = self.prepare_targets(targets, test=test)

        calib = target_vars['calib']
        pad_size = target_vars['pad_size']
        img_size = target_vars['img_size']

        # evaluate disentangling iou for each components in [loc/dim/ori]
        if self.eval_dis_iou:
            dis_ious = self.evaluate_det_3d(target_vars, pred_reg)
        else:
            dis_ious = None

        # evaluate the accuracy of predicted depths
        if self.eval_depth:
            dep_errs = self.evaluate_dep_3d(target_vars, pred_reg)
        else:
            dep_errs = None

        heatmap = nms_hm(pred_hmp)
        vis_sources = {'heat_map': pred_hmp.clone()}

        # select top-k of the predicted heatmap
        scores, indexes, classes, ys, xs = select_topk(heatmap, K=self.max_detection)
        pred_reg_pois = select_point_of_interest(batch_size, indexes, pred_reg).view(-1, pred_reg.shape[1])
        pred_box_pts = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)

        # thresholding with score
        scores = scores.view(-1)
        valid_mask = scores >= self.det_threshold

        # no valid predictions
        if valid_mask.sum() == 0:
            result = scores.new_zeros(0, 14)
            vis_sources['keypoints'] = scores.new_zeros(0, 20)
            vis_sources['project_center'] = scores.new_zeros(0, 2)
            eval_utils = {
                'dis_ious': dis_ious,
                'dep_errs': dep_errs,
                'vis_scores': scores.new_zeros(0),
                'uncertainty_conf': scores.new_zeros(0),
                'estimated_depth_error': scores.new_zeros(0),
            }
            return result, eval_utils, vis_sources

        scores = scores[valid_mask]
        classes = classes.view(-1)[valid_mask]
        pred_box_pts = pred_box_pts[valid_mask]
        pred_reg_pois = pred_reg_pois[valid_mask]

        pred_reg_2d = F.relu(pred_reg_pois[:, self.kvper('dim_2d')])
        pred_dim_offset_3d = pred_reg_pois[:, self.kvper('dim_3d')]
        pred_offset_3d = pred_reg_pois[:, self.kvper('offset_3d')]
        pred_orient_3d = torch.cat(
            (pred_reg_pois[:, self.kvper('orient_cls')], pred_reg_pois[:, self.kvper('orient_ofs')]), dim=1
        )
        vis_sources['project_center'] = pred_box_pts + pred_offset_3d

        pred_box2d = self.coder.decode_box2d_fcos(pred_box_pts, pred_reg_2d, pad_size, img_size)
        pred_dim3d = self.coder.decode_dim3d(classes, pred_dim_offset_3d)

        if self.enable_direct_depth:
            pred_depths_offset = pred_reg_pois[:, self.kvper('depth')].squeeze(-1)
            pred_direct_depths = self.coder.decode_depth(pred_depths_offset)

        if self.enable_uncert_depth:
            pred_direct_uncertainty = pred_reg_pois[:, self.kvper('depth_uncertainty')].exp()
            vis_sources['depth_uncertainty'] = pred_reg[:, self.kvper('depth_uncertainty'), ...].squeeze(1)

        if self.enable_keypts_corner:
            pred_kpt_offset = pred_reg_pois[:, self.kvper('corner_offset')]
            pred_kpt_offset = pred_kpt_offset.view(-1, 10, 2)
            # solve depth from estimated key-points
            pred_keypoints_depths = self.coder.decode_depth_from_kpts_batch(pred_kpt_offset, pred_dim3d, calib)
            vis_sources['keypoints'] = pred_kpt_offset

        if self.enable_uncert_corner:
            pred_kpt_uncertainty = pred_reg_pois[:, self.kvper('corner_uncertainty')].exp()

        estimated_depth_error = None

        if self.output_depth == 'direct':
            pred_depths = pred_direct_depths
            if self.enable_uncert_depth:
                estimated_depth_error = pred_direct_uncertainty.squeeze(dim=1)
        elif self.output_depth.find('keypoints') >= 0:
            if self.output_depth == 'keypoints_avg':
                pred_depths = pred_keypoints_depths.mean(dim=1)
                if self.enable_uncert_corner:
                    estimated_depth_error = pred_kpt_uncertainty.mean(dim=1)
            elif self.output_depth == 'keypoints_center':
                pred_depths = pred_keypoints_depths[:, 0]
                if self.enable_uncert_corner:
                    estimated_depth_error = pred_kpt_uncertainty[:, 0]
            elif self.output_depth == 'keypoints_02':
                pred_depths = pred_keypoints_depths[:, 1]
                if self.enable_uncert_corner:
                    estimated_depth_error = pred_kpt_uncertainty[:, 1]
            elif self.output_depth == 'keypoints_13':
                pred_depths = pred_keypoints_depths[:, 2]
                if self.enable_uncert_corner:
                    estimated_depth_error = pred_kpt_uncertainty[:, 2]
            else:
                raise ValueError

        # hard ensemble, soft ensemble and simple average
        elif self.output_depth in ['hard', 'soft', 'mean', 'oracle']:
            if self.enable_direct_depth and self.enable_uncert_depth:
                pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
                pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_kpt_uncertainty), dim=1)
            else:
                pred_combined_depths = pred_keypoints_depths.clone()
                pred_combined_uncertainty = pred_kpt_uncertainty.clone()

            depth_weights = 1 / pred_combined_uncertainty
            vis_sources['min_uncertainty'] = depth_weights.argmax(dim=1)

            if self.output_depth == 'hard':
                pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]),
                                                   depth_weights.argmax(dim=1)]
                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.min(dim=1).values
            elif self.output_depth == 'soft':
                depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
                pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)
                # the uncertainty after combination
                estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)
            elif self.output_depth == 'mean':
                pred_depths = pred_combined_depths.mean(dim=1)
                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.mean(dim=1)
            # the best estimator is always selected
            elif self.output_depth == 'oracle':
                pred_depths, estimated_depth_error = self.get_oracle_depths(
                    pred_box2d, classes, pred_combined_depths, pred_combined_uncertainty, targets[0]
                )

        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
        pred_loc3d = self.coder.decode_loc3d(pred_box_pts, pred_offset_3d, pred_depths, calib, pad_size, batch_idxs)
        pred_rty3d, pred_alpha = self.coder.decode_orient(pred_orient_3d, pred_loc3d)

        scores = scores.view(-1, 1)
        classes = classes.view(-1, 1)
        pred_loc3d[:, 1] += pred_dim3d[:, 1] / 2
        pred_alpha = pred_alpha.view(-1, 1)
        pred_rty3d = pred_rty3d.view(-1, 1)
        pred_dim3d = pred_dim3d.roll(shifts=-1, dims=1)  # switch dim back to [h,w,l]

        # the uncertainty of depth estimation can reflect the confidence for 3D object detection
        vis_scores = scores.clone()
        if self.enable_uncert_confidence and estimated_depth_error is not None:
            uncertainty_conf = 1 - torch.clamp(estimated_depth_error, min=0.01, max=1)
            scores = scores * uncertainty_conf.view(-1, 1)
        else:
            uncertainty_conf, estimated_depth_error = None, None

        # kitti output format
        result = torch.cat([classes, pred_alpha, pred_box2d, pred_dim3d, pred_loc3d, pred_rty3d, scores], dim=1)

        eval_utils = {
            'dis_ious': dis_ious,
            'dep_errs': dep_errs,
            'uncertainty_conf': uncertainty_conf,
            'estimated_depth_error': estimated_depth_error,
            'vis_scores': vis_scores,
        }
        return result, eval_utils, vis_sources

    def get_oracle_depths(self, pred_bboxes, pred_clses, pred_combined_depths, pred_combined_uncertainty, target):
        calib = target.get_field('calib')
        pad_size = target.get_field('pad_size')
        pad_w, pad_h = pad_size
        valid_mask = target.get_field('tgt_fill_mask').bool()
        gt_num = valid_mask.sum()
        gt_class = target.get_field('tgt_clsids')[valid_mask]
        gt_box2d = target.get_field('tgt_box2ds')[valid_mask]
        gt_loc3d = target.get_field('tgt_loc3ds')[valid_mask]
        gt_depth = gt_loc3d[:, -1]
        gt_box2d_center = (gt_box2d[:, :2] + gt_box2d[:, 2:]) / 2

        iou_thresh = 0.5

        # initialize with the average values
        oracle_depth = pred_combined_depths.mean(dim=1)
        estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

        for i in range(pred_bboxes.shape[0]):
            # find the corresponding object bounding boxes
            box2d = pred_bboxes[i]
            box2d_center = (box2d[:2] + box2d[2:]) / 2
            img_dis = torch.sum((box2d_center.reshape(1, 2) - gt_box2d_center)**2, dim=1)
            equal_class_mask = gt_class == pred_clses[i]
            img_dis[~equal_class_mask] = 9999
            near_idx = torch.argmin(img_dis)
            # iou 2d
            iou_2d = box_iou(box2d.detach().cpu().numpy(), gt_box2d[near_idx].detach().cpu().numpy())

            if iou_2d < iou_thresh:
                # match failed, simply choose the default average
                continue
            else:
                estimator_index = torch.argmin(torch.abs(pred_combined_depths[i] - gt_depth[near_idx]))
                oracle_depth[i] = pred_combined_depths[i, estimator_index]
                estimated_depth_error[i] = pred_combined_uncertainty[i, estimator_index]

        return oracle_depth, estimated_depth_error

    def evaluate_dep_3d(self, targets, pred_reg):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch_size, channel = pred_reg.shape[:2]

        # 1. extract prediction in points of interest
        target_points = targets['tgt_center_2ds'].float()
        pred_reg_pois = select_point_of_interest(batch_size, target_points, pred_reg)

        pred_reg_pois = pred_reg_pois.view(-1, channel)
        tgt_fill_mask = targets['tgt_fill_mask'].view(-1).bool()
        pred_reg_pois = pred_reg_pois[tgt_fill_mask]
        target_points = target_points[0][tgt_fill_mask]

        # depth predictions
        pred_depths_offset = pred_reg_pois[:, self.kvper('depth')]
        pred_kpt_offset = pred_reg_pois[:, self.kvper('corner_offset')]

        pred_direct_uncertainty = pred_reg_pois[:, self.kvper('depth_uncertainty')].exp()
        pred_kpt_uncertainty = pred_reg_pois[:, self.kvper('corner_uncertainty')].exp()

        # dimension predictions
        target_clsids = targets['tgt_clsids'].view(-1)[tgt_fill_mask]
        pred_dim_offset_3d = pred_reg_pois[:, self.kvper('dim_3d')]
        pred_dim3d = self.coder.decode_dim3d(
            target_clsids,
            pred_dim_offset_3d,
        )
        # direct
        pred_direct_depths = self.coder.decode_depth(pred_depths_offset.squeeze(-1))
        # three depths from keypoints
        pred_keypoints_depths = self.coder.decode_depth_from_kpts_batch(
            pred_kpt_offset.view(-1, 10, 2), pred_dim3d, targets['calib']
        )
        # combined uncertainty
        pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_kpt_uncertainty), dim=1)
        pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

        # min-uncertainty
        pred_uncertainty_min_depth = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]),
                                                          pred_combined_uncertainty.argmin(dim=1)]
        pred_uncertainty_weights = 1 / pred_combined_uncertainty
        pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
        pred_uncertainty_softmax_depth = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)

        # 3. get ground-truth
        target_loc3d = targets['tgt_loc3ds'].view(-1, 3)[tgt_fill_mask]
        target_depth = target_loc3d[:, -1]

        # abs error
        pred_combined_error = (pred_combined_depths - target_depth[:, None]).abs()
        pred_uncertainty_min_error = (pred_uncertainty_min_depth - target_depth).abs()
        pred_uncertainty_softmax_error = (pred_uncertainty_softmax_depth - target_depth).abs()
        pred_direct_error = pred_combined_error[:, 0]
        pred_keypoints_error = pred_combined_error[:, 1:]

        pred_mean_depth = pred_combined_depths.mean(dim=1)
        pred_mean_error = (pred_mean_depth - target_depth).abs()
        # upper-bound
        pred_min_error = pred_combined_error.min(dim=1)[0]

        pred_errors = {
            'direct': pred_direct_error,
            'direct_sigma': pred_direct_uncertainty[:, 0],
            'keypoint_center': pred_keypoints_error[:, 0],
            'keypoint_center_sigma': pred_kpt_uncertainty[:, 0],
            'keypoint_02': pred_keypoints_error[:, 1],
            'keypoint_02_sigma': pred_kpt_uncertainty[:, 1],
            'keypoint_13': pred_keypoints_error[:, 2],
            'keypoint_13_sigma': pred_kpt_uncertainty[:, 2],
            'sigma_min': pred_uncertainty_min_error,
            'sigma_weighted': pred_uncertainty_softmax_error,
            'mean': pred_mean_error,
            'min': pred_min_error,
            'target': target_depth,
        }
        return pred_errors

    def evaluate_det_3d(self, targets, pred_reg):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch_size, channel = pred_reg.shape[:2]

        # 1. extract prediction in points of interest
        target_points = targets['tgt_center_2ds'].float()
        pred_reg_pois = select_point_of_interest(batch_size, target_points, pred_reg)

        # 2. get needed predictions
        pred_reg_pois = pred_reg_pois.view(-1, channel)
        tgt_fill_mask = targets['tgt_fill_mask'].view(-1).bool()
        pred_reg_pois = pred_reg_pois[tgt_fill_mask]
        target_points = target_points[0][tgt_fill_mask]

        pred_offset_3d = pred_reg_pois[:, self.kvper('offset_3d')]
        pred_orient_3d = torch.cat(
            (pred_reg_pois[:, self.kvper('orient_cls')], pred_reg_pois[:, self.kvper('orient_ofs')]), dim=1
        )
        pred_dim_offset_3d = pred_reg_pois[:, self.kvper('dim_3d')]
        pred_kpt_offset = pred_reg_pois[:, self.kvper('corner_offset')].view(-1, 10, 2)

        # 3. get ground-truth
        target_clsids = targets['tgt_clsids'].view(-1)[tgt_fill_mask]
        target_offset_3D = targets['tgt_offset_3ds'].view(-1, 2)[tgt_fill_mask]
        target_loc3d = targets['tgt_loc3ds'].view(-1, 3)[tgt_fill_mask]
        target_dim3d = targets['tgt_dim3ds'].view(-1, 3)[tgt_fill_mask]
        target_rty3d = targets['tgt_rty3ds'].view(-1)[tgt_fill_mask]

        target_depth = target_loc3d[:, -1]

        # 4. decode prediction
        pred_dim3d = self.coder.decode_dim3d(target_clsids, pred_dim_offset_3d)

        pred_depths_offset = pred_reg_pois[:, self.kvper('depth')].squeeze(-1)
        if self.output_depth == 'direct':
            pred_depths = self.coder.decode_depth(pred_depths_offset)
        elif self.output_depth == 'keypoints':
            pred_depths = self.coder.decode_depth_from_kpts(
                pred_offset_3d, pred_kpt_offset, pred_dim3d, targets['calib']
            )
            pred_uncertainty = pred_reg_pois[:, self.kvper('corner_uncertainty')].exp()
            pred_depths = pred_depths[torch.arange(pred_depths.shape[0]), pred_uncertainty.argmin(dim=1)]
        elif self.output_depth == 'combine':
            pred_direct_depths = self.coder.decode_depth(pred_depths_offset)
            pred_keypoints_depths = self.coder.decode_depth_from_kpts(
                pred_offset_3d, pred_kpt_offset, pred_dim3d, targets['calib']
            )
            pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
            pred_direct_uncertainty = pred_reg_pois[:, self.kvper('depth_uncertainty')].exp()
            pred_kpt_uncertainty = pred_reg_pois[:, self.kvper('corner_uncertainty')].exp()
            pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_kpt_uncertainty), dim=1)
            pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]),
                                               pred_combined_uncertainty.argmin(dim=1)]

        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
        pred_loc3d_offset = self.coder.decode_loc3d(
            target_points, pred_offset_3d, target_depth, targets['calib'], targets["pad_size"], batch_idxs
        )
        pred_loc3d_depth = self.coder.decode_loc3d(
            target_points, target_offset_3D, pred_depths, targets['calib'], targets["pad_size"], batch_idxs
        )
        pred_loc3d = self.coder.decode_loc3d(
            target_points, pred_offset_3d, pred_depths, targets['calib'], targets["pad_size"], batch_idxs
        )
        pred_rty3d, _ = self.coder.decode_orient(pred_orient_3d, target_loc3d)
        full_pred_rotys, _ = self.coder.decode_orient(pred_orient_3d, pred_loc3d)

        # fully predicted
        pred_box3d = torch.cat((pred_loc3d, pred_dim3d, full_pred_rotys[:, None]), dim=1)
        # ground-truth
        target_box3d = torch.cat((target_loc3d, target_dim3d, target_rty3d[:, None]), dim=1)
        # disentangling
        offset_box3d = torch.cat((pred_loc3d_offset, target_dim3d, target_rty3d[:, None]), dim=1)
        depth_box3d = torch.cat((pred_loc3d_depth, target_dim3d, target_rty3d[:, None]), dim=1)
        dim_box3d = torch.cat((target_loc3d, pred_dim3d, target_rty3d[:, None]), dim=1)
        orient_box3d = torch.cat((target_loc3d, target_dim3d, pred_rty3d[:, None]), dim=1)

        # 6. compute 3D IoU
        pred_IoU = get_iou3d(pred_box3d, target_box3d)
        offset_IoU = get_iou3d(offset_box3d, target_box3d)
        depth_IoU = get_iou3d(depth_box3d, target_box3d)
        dims_IoU = get_iou3d(dim_box3d, target_box3d)
        orien_IoU = get_iou3d(orient_box3d, target_box3d)
        output = dict(
            pred_IoU=pred_IoU, offset_IoU=offset_IoU, depth_IoU=depth_IoU, dims_IoU=dims_IoU, orien_IoU=orien_IoU
        )
        return output


def get_iou3d(source_boxes, target_boxes):
    num_query = target_boxes.shape[0]

    # compute overlap along y axis
    min_h_a = -(source_boxes[:, 1] + source_boxes[:, 4] / 2)
    max_h_a = -(source_boxes[:, 1] - source_boxes[:, 4] / 2)
    min_h_b = -(target_boxes[:, 1] + target_boxes[:, 4] / 2)
    max_h_b = -(target_boxes[:, 1] - target_boxes[:, 4] / 2)

    # overlap in height
    h_max_of_min = torch.max(min_h_a, min_h_b)
    h_min_of_max = torch.min(max_h_a, max_h_b)
    h_overlap = (h_min_of_max - h_max_of_min).clamp_(min=0)

    # volumes of bboxes
    source_volumes = source_boxes[:, 3] * source_boxes[:, 4] * source_boxes[:, 5]
    target_volumes = target_boxes[:, 3] * target_boxes[:, 4] * target_boxes[:, 5]

    # derive x y l w alpha
    source_boxes = source_boxes[:, [0, 2, 3, 5, 6]]
    target_boxes = target_boxes[:, [0, 2, 3, 5, 6]]

    # convert bboxes to corners
    source_corners = get_corners(source_boxes)
    target_corners = get_corners(target_boxes)
    iou_3d = source_boxes.new_zeros(num_query)

    for i in range(num_query):
        source_polygon = Polygon(source_corners[i])
        target_polygon = Polygon(target_corners[i])
        overlap = source_polygon.intersection(target_polygon).area

        overlap_3d = overlap * h_overlap[i]
        union3d = source_polygon.area * (max_h_a[0] - min_h_a[0]) + \
                  target_polygon.area * (max_h_b[i] - min_h_b[i]) - overlap_3d
        iou_3d[i] = overlap_3d / union3d

    return iou_3d
