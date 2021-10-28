import torch
import torch.nn as nn
"""
References:
    A Intriguing Failing of Convolution Neural Networks and the CoordConv Solution, NeurIPS'18
"""


class CoordConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, bias=False, with_radius=False):
        super(CoordConv2d, self).__init__()
        inplanes = inplanes + 2
        if with_radius:
            inplanes = inplanes + 1
        self.add_coord = AddCoords(with_radius=with_radius)
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias
        )

    def forward(self, x):
        ret = self.add_coord(x)
        ret = self.conv(ret)
        return ret


class AddCoords(nn.Module):

    def __init__(self, with_radius=False):
        super(AddCoords, self).__init__()
        self.with_radius = with_radius

    def forward(self, feats):
        batch, _, x_dim, y_dim = feats.size()

        xx_chans = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_chans = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_chans = xx_chans.float() / (x_dim - 1)
        yy_chans = yy_chans.float() / (y_dim - 1)
        xx_chans = xx_chans * 2 - 1
        yy_chans = yy_chans * 2 - 1

        xx_chans = xx_chans.repeat(batch, 1, 1, 1).transpose(2, 3)
        yy_chans = yy_chans.repeat(batch, 1, 1, 1).transpose(2, 3)

        feats_update = torch.cat([feats, xx_chans.type_as(feats), yy_chans.type_as(feats)], dim=1)

        if self.with_radius:
            rr = torch.sqrt(torch.pow(xx_chans.type_as(feats) - 0.5, 2) + torch.pow(yy_chans.type_as(feats) - 0.5, 2))
            feats_update = torch.cat([feats_update, rr], dim=1)

        return feats_update


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 32, 32)
    coord_conv = CoordConv2d(inplanes=3, planes=6)
    print(coord_conv(inputs).shape)