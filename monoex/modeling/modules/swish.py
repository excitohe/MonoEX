import torch.nn as nn
"""
References:
    Swish: a Self-Gated Activation Function, arXiv'17
"""


def swish(x, inplace: bool = False):
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)