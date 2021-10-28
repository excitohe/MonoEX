import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from monoex.layers import DeformConv2dPack
from monoex.modeling.utils import fill_up_weights, get_norm

from .build import BACKBONE_REGISTRY


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, norm="BN", stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation
        )
        self.bn1 = get_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, norm="BN", stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm, bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = get_norm(norm, bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, norm="BN", stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32

        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm, bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality
        )
        self.bn2 = get_norm(norm, bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, residual, norm="BN"):
        super(Root, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):

    def __init__(
        self,
        levels,
        block,
        inplanes,
        planes,
        norm="BN",
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * planes
        if level_root:
            root_dim += inplanes
        if levels == 1:
            self.tree1 = block(inplanes, planes, norm, stride, dilation=dilation)
            self.tree2 = block(planes, planes, norm, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                inplanes,
                planes,
                norm=norm,
                stride=stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                planes,
                planes,
                norm=norm,
                root_dim=root_dim + planes,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual
            )
        if levels == 1:
            self.root = Root(root_dim, planes, root_kernel_size, root_residual, norm=norm)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if inplanes != planes:
            self.project = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                get_norm(norm, planes),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):

    def __init__(
        self, levels, channels, norm="BN", num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False
    ):
        super(DLA, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.level_length = len(levels)

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), get_norm(norm, channels[0]),
            nn.ReLU(inplace=True)
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], norm)
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], norm, stride=2)
        self.level2 = Tree(
            levels[2], block, channels[1], channels[2], norm, stride=2, level_root=False, root_residual=residual_root
        )
        self.level3 = Tree(
            levels[3], block, channels[2], channels[3], norm, stride=2, level_root=True, root_residual=residual_root
        )
        self.level4 = Tree(
            levels[4], block, channels[3], channels[4], norm, stride=2, level_root=True, root_residual=residual_root
        )
        self.level5 = Tree(
            levels[5], block, channels[4], channels[5], norm, stride=2, level_root=True, root_residual=residual_root
        )

    def _make_conv_level(self, inplanes, planes, convs, norm, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    ),
                    get_norm(norm, planes),
                    nn.ReLU(inplace=True)
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla46_c')
    return model


def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla46x_c')
    return model


def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=False, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla60')
    return model


def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla60x')
    return model


def dla102(pretrained=False, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102')
    return model


def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102x')
    return model


def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102x2')
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla169')
    return model


class NaivesConv(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, norm="BN", bias=True):
        super(NaivesConv, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias
        )
        self.bn = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeformConv(nn.Module):

    def __init__(self, inplanes, planes, norm="BN"):
        super(DeformConv, self).__init__()
        self.dcn = nn.Sequential(
            DeformConv2dPack(inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1),
            get_norm(norm, planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dcn(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, inplane_list, plane, up_factor, use_dcn=True, norm="BN"):
        super(IDAUp, self).__init__()
        self.inplane_list = inplane_list
        self.plane = plane

        for i in range(1, len(inplane_list)):
            inplane = inplane_list[i]
            f = int(up_factor[i])
            if use_dcn:
                proj = DeformConv(inplane, plane, norm)
                node = DeformConv(plane, plane, norm)
            else:
                proj = NaivesConv(inplane, plane, norm=norm, bias=False)
                node = NaivesConv(plane, plane, norm=norm, bias=False)

            up = nn.ConvTranspose2d(
                plane, plane, kernel_size=f * 2, stride=f, padding=f // 2, output_padding=0, groups=plane, bias=False
            )
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self, startp, channels, scales, inplanes=None, use_dcn=True, norm="BN"):
        super(DLAUp, self).__init__()

        self.startp = startp
        if inplanes is None:
            inplanes = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(inplanes[j:], channels[j], scales[j:] // scales[j], use_dcn, norm))
            scales[j + 1:] = scales[j]
            inplanes[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLACompose(nn.Module):

    def __init__(self, base_name, pretrained, down_ratio, last_level, use_dcn, norm):
        super(DLACompose, self).__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level

        self.base = globals()[base_name](pretrained=pretrained, norm=norm)

        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            startp=self.first_level, channels=channels[self.first_level:], scales=scales, use_dcn=use_dcn, norm=norm
        )
        self.planes = channels[self.first_level]
        up_scales = [2**i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(channels[self.first_level:self.last_level], self.planes, up_factor=up_scales, norm=norm)
        self.out_channels = self.planes

    def forward(self, x):
        # x: list of features with stride = 1, 2, 4, 8, 16, 32
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]


@BACKBONE_REGISTRY.register()
def build_dlanet_backbone(cfg):
    model = DLACompose(
        base_name=cfg.MODEL.BACKBONE.CONV_BODY,
        pretrained=cfg.MODEL.PRETRAIN,
        down_ratio=cfg.MODEL.BACKBONE.DOWN_RATIO,
        last_level=5,
        use_dcn=cfg.MODEL.BACKBONE.USE_DCN,
        norm=cfg.MODEL.BACKBONE.NORM,
    )
    return model
