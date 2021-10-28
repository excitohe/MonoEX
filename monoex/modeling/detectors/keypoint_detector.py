import torch.nn as nn
from monoex.modeling.backbones import build_backbone
from monoex.modeling.heads import build_head
from monoex.structures.image_list import to_image_list

from .build import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class KeypointDetector(nn.Module):
    """
    Generalized architecture for keypoint based detector.
    main parts:
    - backbone
    - head
    """

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg, self.backbone.out_channels)

        self.test = cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if self.training:
            loss_dict, logs_dict = self.head(features, targets)
            return loss_dict, logs_dict
        else:
            results, eval_utils, vis_preds = self.head(features, targets, test=self.test)
            return results, eval_utils, vis_preds
