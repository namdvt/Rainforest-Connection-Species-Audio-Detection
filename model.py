import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, conv1x1, Bottleneck
import matplotlib.pyplot as plt
import torch.nn.functional as F
import timm
from prettytable import PrettyTable


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
    print(table)
    print(f"Total Trainable Params: {total_params}")
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
        # self.backbone = models.resnet50(pretrained=True)
        self.backbone = timm.create_model(backbone, pretrained=True)
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3]) # r50
        # self.backbone = ResNet(BasicBlock, [2, 2, 2, 2]) #r18
        # self.backbone = ResNet(BasicBlock, [1, 1, 1, 1]) #r9
        self.classify = nn.Sequential(
            Linear(1000, 512),
            nn.Linear(512, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classify(x)

        return x


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1, dropout=0):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for c in self.modules():
            if isinstance(c, nn.Conv1d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, num_blocks=1, dropout=0):
        super(SubBlock, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        self.conv = nn.Sequential()
        self.conv.add_module('first_conv', Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout))
        for i in range(num_blocks - 1):
            self.conv.add_module(str(i), Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout))
        self.conv.add_module('conv1d', nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.conv.add_module('bn', nn.BatchNorm1d(out_channels))
        self.conv.add_module('dropout', nn.Dropout(p=dropout))
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x_shortcut = x.clone()
        x_shortcut = self.conv1x1(x_shortcut)

        x = self.conv(x)
        x += x_shortcut
        x = self.bn_relu(x)

        return x


class MainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0):
        super(MainBlock, self).__init__()
        self.conv = nn.Sequential(
            SubBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout, num_blocks=2),
            SubBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dropout=dropout, num_blocks=2),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Model1DCNN(nn.Module):
    def __init__(self):
        super(Model1DCNN, self).__init__()
        self.blocks = nn.Sequential(
            Conv1d(1, 128, kernel_size=3, stride=1),
            MainBlock(128, 128, kernel_size=3, padding=1, dropout=0),
            nn.MaxPool1d(3),
            MainBlock(128, 256, kernel_size=3, padding=1, dropout=0),
            # nn.MaxPool1d(3),
            # MainBlock(128, 512, kernel_size=3, padding=1, dropout=0),
            # nn.MaxPool1d(3),
            # MainBlock(512, 1024, kernel_size=3, padding=1, dropout=0),
        )
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            # Linear(1024, 512),
            Linear(256, 128),
            nn.Linear(128, 24),
            nn.Sigmoid()
        )

    def forward(self, x, classify=True):
        x = self.blocks(x)
        if classify:
            x = self.classify(x)
        return x


class Model1DCNNSimple(nn.Module):
    def __init__(self):
        super(Model1DCNNSimple, self).__init__()
        self.conv = nn.Sequential(
            Conv1d(1, 32, kernel_size=3, padding=1),
            Conv1d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            Conv1d(64, 64, kernel_size=3, padding=1),
            Conv1d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            Conv1d(128, 128, kernel_size=3, padding=1),
            Conv1d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            Conv1d(256, 256, kernel_size=3, padding=1),
            Conv1d(256, 512, kernel_size=3, padding=1),
        )

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            Linear(512, 256),
            Linear(256, 128),
            nn.Linear(128, 24),
            nn.Sigmoid()
        )

    def forward(self, x, classify=True):
        x = self.conv(x)
        if classify:
            x = self.classify(x)
        return x


if __name__ == '__main__':
    sample = torch.ones((2, 1, 128))
    # net = SubBlock(in_channels=2, out_channels=4, num_blocks=1, dropout=0.2)
    net = Model1DCNNSimple()
    net(sample)

    print()
