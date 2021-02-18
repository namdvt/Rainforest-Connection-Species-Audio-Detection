import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, conv1x1, Bottleneck
import matplotlib.pyplot as plt
import torch.nn.functional as F
import timm
from prettytable import PrettyTable
from test_sincnet.dnn_models import *


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        # backbone
        if backbone == 'selecsls42b':
            from model.backbones.selecsls42b import create_model
            self.backbone = create_model()
        elif backbone == 'legacy_seresnet34':
            from model.backbones.legacy_seresnet34 import create_model
            self.backbone = create_model()
        elif backbone == 'swsl_resnext50_32x4d':
            from model.backbones.swsl_resnext50_32x4d import create_model
            self.backbone = create_model()
        else:
            self.backbone = timm.create_model(backbone, pretrained=True)
        
        self.classify = nn.Sequential(
            Linear(1000, 512),
            nn.Linear(512, 24),
            # nn.Sigmoid()
        )

    def forward(self, x, is_sigmoid=True):
        x = self.backbone(x)
        x = self.classify(x)
        if is_sigmoid:
            x = torch.sigmoid(x)

        return x


# if __name__ == '__main__':
#     import numpy as np
#     sr = 48000
#     # sample = torch.tensor(sample).unsqueeze(0).unsqueeze(0)
#     sample = torch.zeros((2,1,48000))
#     model = Model1d(sample_rate=sr)
#     model(sample)
#     print()
