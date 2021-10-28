import torch.nn as nn
from monoex.modeling.heads import HEAD_REGISTRY

from . import GUPPredictor


@HEAD_REGISTRY.register()
class GUPHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(GUPHead, self).__init__()

        self.predictor = GUPPredictor(cfg, in_channels)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features, targets)