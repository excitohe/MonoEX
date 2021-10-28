import torch.nn as nn
from monoex.modeling.heads import HEAD_REGISTRY

from . import MonoFlexEvaluator, MonoFlexPredictor, MonoFlexProcessor


@HEAD_REGISTRY.register()
class MonoFlexHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MonoFlexHead, self).__init__()

        self.predictor = MonoFlexPredictor(cfg, in_channels)
        self.evaluator = MonoFlexEvaluator(cfg)
        self.processor = MonoFlexProcessor(cfg)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features, targets)
        if self.training:
            loss_dict, log_loss_dict = self.evaluator(x, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, vis_preds = self.processor(x, targets, test=test, features=features)
            return result, eval_utils, vis_preds
