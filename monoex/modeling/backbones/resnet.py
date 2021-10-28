import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from monoex.modeling.utils import get_norm

from .build import BACKBONE_REGISTRY


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm="BN", stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = get_norm(norm, planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm="BN", stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = get_norm(norm, planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = get_norm(norm, planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = get_norm(norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    planes = [64, 128, 256, 512]

    def __init__(
        self,
        block,
        layers,
        num_stages,
        norm="BN",
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(-1, 0, 1, 2, 3),
        freeze_at=-1,
        norm_eval=True
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.norm = norm
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.freeze_at = freeze_at
        self.norm_eval = norm_eval
        assert max(out_indices) < num_stages

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = get_norm(self.norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(num_stages):
            setattr(
                self, f"layer{i+1}",
                self._make_layer(block, self.planes[i], layers[i], stride=self.strides[i], dilation=self.dilations[i])
            )

        self.init_weights()
        self.train()

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if -1 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)
        for i in range(self.num_stages):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                get_norm(self.norm, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        """ Freeze both BatchNorm and SyncBatchNorm
        """
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()

    def freeze_stages(self):
        if self.freeze_at >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        for i in range(1, self.freeze_at + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """ Convert the model into training mode 
            while keep normalization layer freezed.
        """
        super(ResNet, self).train(mode)
        self.freeze_stages()
        if mode and self.norm_eval:
            self.freeze_bn()


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg):
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    arch_settings = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }

    conv_body = cfg.MODEL.BACKBONE.CONV_BODY
    if conv_body not in arch_settings:
        raise KeyError(f'invalid conv_body {conv_body} for resnet')
    stage_block, stage_layer = arch_settings[conv_body]

    model = ResNet(
        stage_block,
        stage_layer,
        num_stages=cfg.MODEL.BACKBONE.RESNETS.NUM_STAGES,
        norm=cfg.MODEL.BACKBONE.NORM,
        strides=cfg.MODEL.BACKBONE.RESNETS.STRIDES,
        dilations=cfg.MODEL.BACKBONE.RESNETS.DILATIONS,
        out_indices=cfg.MODEL.BACKBONE.RESNETS.OUT_INDICES,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,
        norm_eval=cfg.MODEL.BACKBONE.RESNETS.NORM_EVAL,
    )

    pretrained = cfg.MODEL.PRETRAIN
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[conv_body], model_dir='.'), strict=False)
    return model
