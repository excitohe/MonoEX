import torch.nn as nn
from monoex.modeling.heads import HEAD_REGISTRY

from . import SMOKEPredictor, SMOKEEvaluator, SMOKEProcessor


@HEAD_REGISTRY.register()
class SMOKEHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.predictor = SMOKEPredictor(cfg, in_channels)
        self.evaluator = SMOKEEvaluator(cfg)
        self.processor = SMOKEProcessor(cfg)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features)
        if self.training:
            loss_dict, log_loss_dict = self.evaluator(x, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, vis_preds = self.processor(x, targets, test=test, features=features)
            return result, eval_utils, vis_preds