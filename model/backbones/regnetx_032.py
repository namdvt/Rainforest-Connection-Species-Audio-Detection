import numpy as np
import torch.nn as nn

from timm.models.layers import ClassifierHead, AvgPool2dSame, ConvBnAct, SEModule, DropPath

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, bottleneck_ratio=1, group_width=1, se_ratio=0.25,
                 downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvBnAct(
            bottleneck_chs, bottleneck_chs, kernel_size=3, stride=1, dilation=dilation,
            groups=groups, **cargs)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, reduction_channels=se_channels)
        else:
            self.se = None
        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


def downsample_conv(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    return ConvBnAct(
        in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, norm_layer=norm_layer, act_layer=None)


def downsample_avg(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[
        pool, ConvBnAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, act_layer=None)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio, group_width,
                 block_fn=Bottleneck, se_ratio=0., drop_path_rate=None, drop_block=None):
        super(RegStage, self).__init__()
        block_kwargs = {}  # FIXME setup to pass various aa, norm, act layer common args
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation if i == 0 else dilation
            drop_path = DropPath(drop_path_rate[i]) if drop_path_rate is not None else None
            if (block_in_chs != out_chs) or (block_stride != 1):
                proj_block = downsample_conv(block_in_chs, out_chs, 1, block_stride, block_dilation)
            else:
                proj_block = None

            name = "b{}".format(i + 1)
            self.add_module(
                name, block_fn(
                    block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio,
                    downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.,
                 drop_path_rate=0., zero_init_last_bn=True):
        super().__init__()
        # TODO add drop block, drop path, anti-aliasing, custom bn/act args
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)

        # Construct the stem
        stem_width = cfg['stem_width']
        self.stem = ConvBnAct(in_chans, stem_width, 3, stride=2)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 1
        stage_params = self._get_stage_params(cfg, output_stride=output_stride, drop_path_rate=drop_path_rate)
        se_ratio = cfg['se_ratio']
        for i, stage_args in enumerate(stage_params):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(prev_width, **stage_args, se_ratio=se_ratio))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        self.num_features = prev_width
        self.head = ClassifierHead(
            in_chs=prev_width, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # pooling
        self.maxpool = nn.MaxPool2d((1, 2))
        self.avgpool = nn.AvgPool2d((2, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)

        # Use the same group width, bottleneck mult and stride for each stage
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = [1, 1, 1, 1]
        stage_dilations = [1, 1, 1, 1]
        # net_stride = 2
        # dilation = 1
        # for _ in range(num_stages):
        #     if net_stride >= output_stride:
        #         dilation *= default_stride
        #         stride = 1
        #     else:
        #         stride = default_stride
        #         net_stride *= stride
        #     stage_strides.append(stride)
        #     stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, d), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths, stage_bottle_ratios, stage_groups)
        param_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_width', 'drop_path_rate']
        stage_params = [
            dict(zip(param_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_bottle_ratios, stage_groups,
                stage_dpr)]
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        for idx, block in enumerate(self.children()):
            x = block(x)
            if idx > 0: 
                x = self.avgpool(self.maxpool(x))
        return x


def create_model():
    config = dict(
        se_ratio=0.0, 
        bottle_ratio=1.0,
        stem_width=32,
        w0=88,
        wa=26.31,
        wm=2.25,
        group_w=48,
        depth=25
    )
    return RegNet(config)

if __name__ == "__main__":
    backbone = create_model()