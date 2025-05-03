import torch
import torch.nn as nn
import math
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.pp import fu1, SFF


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        # out = self.bn1(x)
        out = self.relu(x)
        out = self.conv1(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.Res2NetBottleneck = Res2NetBottleneck(mid_ch, mid_ch)
        self.dropout = nn.Dropout2d(0.9)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.Res2NetBottleneck(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class conv_block_nested2(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested2, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.Res2NetBottleneck = Res2NetBottleneck(mid_ch, mid_ch)
        self.dropout = nn.Dropout2d(0.9)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)


        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output
class up(nn.Module):
    def __init__(self, in_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True, norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList(
            [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, 1) for _ in
             range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class CA(nn.Module):
    def __init__(self, inp, reduction):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)

        # mip = max(8, inp // reduction)  论文作者所用
        mip = inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out




class FSRCNN(nn.Module):
    def __init__(self, inchannels):
        super(FSRCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=64, kernel_size=5, stride=1, padding=2,
                      padding_mode='replicate'),
            nn.PReLU()
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
            nn.PReLU()
        )
        self.shrinking2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
            nn.PReLU()
        )

        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PReLU()
        )
        self.mapping2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PReLU()
        )

        self.expanding = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=inchannels, kernel_size=6, stride=4, padding=1)

        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.shrinking(x1)
        x3 = self.mapping(x2) + x2
        x4 = self.shrinking2(x3)
        x5 = self.mapping2(x4) + x4

        x4 = self.expanding(x5)
        x5 = self.deconv(x4)

        return x5


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.ReLU(inplace=True)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

# 3times
class mmNet4(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, drop_rate=0.2):
        super(mmNet4, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 16  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


        self.conv1 = conv1x1(100, 4)
        self.conv1_1_1 = conv1x1(3, 16)
        self.upsample_conv1x1 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0_0 = conv_block_nested(100, 4, 4)

        self.conv0_0 = conv_block_nested(3, filters[0], filters[0])
        self.conv0_0_1 = conv_block_nested(3, filters[0], filters[0])
        self.conv0_0_1_1 = conv_block_nested(4, filters[0], filters[0])
        self.conv0_0_2 = conv_block_nested(filters[0], filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv1_0_1 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv1_0_2 = conv_block_nested(filters[1], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv2_0_1 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv2_0_2 = conv_block_nested(filters[2], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv3_0_1 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv3_0_2 = conv_block_nested(filters[3], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv4_0_1 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv4_0_2 = conv_block_nested(filters[4], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.conv3_1 = conv_block_nested2(filters[4] * 4 + filters[3] * 4, filters[3] * 4, filters[3] * 4)
        self.conv3_2 = conv_block_nested2(filters[3] * 4 + filters[2] * 4, filters[2] * 4, filters[2] * 4)
        self.conv3_3 = conv_block_nested2(filters[2] * 4 + filters[1] * 4, filters[1] * 4, filters[1] * 4)
        self.conv3_4 = conv_block_nested2(filters[1] * 4 + filters[0] * 4, filters[0] * 4, filters[0] * 4)
        self.conv3_5 = conv_block_nested(filters[0] * 4, filters[0], filters[0])
        self.conv3_6 = conv_block_nested2(filters[3] * 4 + filters[4] * 3, filters[3] * 4, filters[3] * 4)
        self.conv3_7 = conv_block_nested2(filters[3] * 4 + filters[2] * 6, filters[2] * 2, filters[2] * 2)
        self.conv3_8 = conv_block_nested2(filters[1] * 6 + filters[2] * 2, filters[1] * 2, filters[1] * 2)
        self.conv3_9 = conv_block_nested2(filters[0] * 6 + filters[1] * 2, filters[0] * 2, filters[0] * 2)
        self.conv3_10 = conv_block_nested(filters[0] * 4, filters[0], filters[0])

        self.conv3_11 = conv_block_nested(filters[0] * 32, filters[0] * 16, filters[0] * 16)
        self.conv3_12 = conv_block_nested(filters[0] * 32, filters[0] * 8, filters[0] * 8)
        self.conv3_13 = conv_block_nested(filters[0] * 16, filters[0] * 4, filters[0] * 4)
        self.conv3_14 = conv_block_nested(filters[0] * 8, filters[0] * 2, filters[0] * 2)
        self.conv3_15 = conv_block_nested(filters[0] * 4, filters[0], filters[0])
        self.conv3_16 = conv_block_nested(filters[0] * 32, filters[0] * 16, filters[0] * 16)
        self.conv3_17 = conv_block_nested(filters[0] * 32, filters[0] * 8, filters[0] * 8)
        self.conv3_18 = conv_block_nested(filters[0] * 16, filters[0] * 4, filters[0] * 4)
        self.conv3_19 = conv_block_nested(filters[0] * 8, filters[0] * 2, filters[0] * 2)
        self.conv3_20 = conv_block_nested(filters[0] * 4, filters[0], filters[0])
        self.Up_6 = up(filters[4] * 2)
        self.Up_7 = up(filters[3] * 4)
        self.Up_8 = up(filters[2] * 2)
        self.Up_9 = up(filters[1] * 2)
        self.Up_6_1 = up(filters[4] * 4)
        self.Up_7_1 = up(filters[3] * 4)
        self.Up_8_1 = up(filters[2] * 4)
        self.Up_9_1 = up(filters[1] * 4)
        self.Up_10 = up(filters[0] * 16)
        self.Up_11 = up(filters[0] * 8)
        self.Up_12 = up(filters[0] * 4)
        self.Up_13 = up(filters[0] * 2)
        self.Up_1 = up(filters[0] * 16)
        self.Up_2 = up(filters[0] * 8)
        self.Up_3 = up(filters[0] * 4)
        self.Up_4 = up(filters[0] * 2)

        # self.Up_9 =nn.ConvTranspose2d(filters[0]*2, filters[0]*2, kernel_size=3, stride=3, padding=0)

        self.upsample_conv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=4, stride=4, padding=0)
        self.upsample_conv2 = nn.ConvTranspose2d(filters[0] * 2, filters[0] * 2, kernel_size=4, stride=4, padding=0)
        self.upsample_conv2_1 = nn.ConvTranspose2d(filters[0] * 2, filters[0] * 1, kernel_size=2, stride=2, padding=0)
        self.upsample_conv3 = nn.ConvTranspose2d(filters[0] * 4, filters[0] * 4, kernel_size=4, stride=4, padding=0)
        self.upsample_conv3_1 = nn.ConvTranspose2d(filters[0] * 4, filters[0] * 2, kernel_size=2, stride=2, padding=0)
        self.upsample_conv4 = nn.ConvTranspose2d(filters[0] * 8, filters[0] * 8, kernel_size=4, stride=4, padding=0)
        self.upsample_conv4_1 = nn.ConvTranspose2d(filters[0] * 8, filters[0] * 4, kernel_size=2, stride=2, padding=0)
        self.upsample_conv5 = nn.ConvTranspose2d(filters[0] * 16, filters[0] * 16, kernel_size=4, stride=4, padding=0)
        self.upsample_conv5_1 = nn.ConvTranspose2d(filters[0] * 16, filters[0] * 8, kernel_size=2, stride=2, padding=0)
        self.upsample_conv6 = nn.ConvTranspose2d(filters[3] * 4, filters[3] * 4, kernel_size=4, stride=4, padding=0)
        self.upsample_conv7 = nn.ConvTranspose2d(filters[2] * 4, filters[2] * 4, kernel_size=4, stride=4, padding=0)
        self.upsample_conv8 = nn.ConvTranspose2d(filters[1] * 4, filters[1] * 4, kernel_size=4, stride=4, padding=0)
        self.upsample_conv9 = nn.ConvTranspose2d(filters[0] * 4, filters[0] * 4, kernel_size=4, stride=4, padding=0)
        self.upsample_conv10 = nn.ConvTranspose2d(3, 24, kernel_size=4, stride=4, padding=0)

        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv_final = nn.Conv2d(filters[0] * 4, 2, kernel_size=1)
        self.conv_final2 = nn.Conv2d(filters[0] * 2, 2, kernel_size=1)

        self.conv1x1_1 = conv1x1(filters[0], filters[2])
        self.conv1x1_2 = conv1x1(filters[1], filters[3])
        self.conv1x1_3 = conv1x1(filters[2], filters[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.SA = CBAMLayer(filters[0])
        self.SA1 = CBAMLayer(filters[0] * 2)
        self.SA2 = CBAMLayer(filters[0] * 4)
        self.SA3 = CBAMLayer(filters[0] * 8)
        self.SA4 = CBAMLayer(filters[0] * 16)

        self.sr = FSRCNN(3)
        self.fu1 = fu1(512)
        self.fu2 = fu1(256)
        self.fu3 = fu1(128)
        self.fu4 = fu1(64)
        self.fu5 = fu1(32)

        self.sff = SFF(256)
        self.sff2 = SFF(128)
        self.sff3 = SFF(64)
        self.sff4 = SFF(32)

        self.downsample_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=78, stride=3, padding=1)

    def forward(self, xA, xB):

        '''A'''
        xA = F.interpolate(xA, size=(64, 64), mode='bilinear', align_corners=False)

        xB_d = F.interpolate(xB, size=(64, 64), mode='bilinear', align_corners=False)
        xA_a = F.interpolate(xA, size=(256, 256), mode='bilinear', align_corners=False)

        xBB = self.sr(xB_d)
        xAA = self.sr(xA)+xA_a
        xBB2 = F.interpolate(xB_d, size=(256, 256), mode='bilinear', align_corners=False)
        # xAAA = F.interpolate(xAAA1, size=(256, 256), mode='bilinear', align_corners=False)
        xAAA = F.interpolate(xAA, size=(64, 64), mode='bilinear', align_corners=False)

        x0_0A = self.conv0_0_1(xA)
        x1_0A = self.conv1_0_1(self.pool(x0_0A))
        x2_0A = self.conv2_0_1(self.pool(x1_0A))
        x3_0A = self.conv3_0_1(self.pool(x2_0A))
        x4_0A = self.conv4_0_1(self.pool(x3_0A))

        xB_1 = self.conv0_0_1(xB_d)
        xB_2 = self.conv1_0_1(self.pool(xB_1))
        xB_3 = self.conv2_0_1(self.pool(xB_2))
        xB_4 = self.conv3_0_1(self.pool(xB_3))
        xB_5 = self.conv4_0_1(self.pool(xB_4))


        '''B'''

        x0_1B = self.conv0_0(xB)
        x1_1B = self.conv1_0(self.pool(x0_1B))
        x2_1B = self.conv2_0(self.pool(x1_1B))
        x3_1B = self.conv3_0(self.pool(x2_1B))
        x4_1B = self.conv4_0(self.pool(x3_1B))

        xA_1 = self.conv0_0(xAA)
        xA_2 = self.conv1_0(self.pool(xA_1))
        xA_3 = self.conv2_0(self.pool(xA_2))
        xA_4 = self.conv3_0(self.pool(xA_3))
        xA_5 = self.conv4_0(self.pool(xA_4))

        '''高到低'''


        # xA4_0 = torch.cat([x4_0A, xB_5, x4_1B, xA_5], 1)
        xA4_0_1 = torch.cat([x4_0A, xB_5], 1)
        xA4_0_2 = torch.cat([self.pool2(x4_1B),self.pool2(xA_5)], 1)
        xA4_0 =self.fu1(xA4_0_1,xA4_0_2)
        # xA4_2 = torch.cat([x3_0A, xB_4, x3_1B, xA_4], 1)
        xA4_2_1 = torch.cat([self.pool2(x3_1B), self.pool2(xA_4)], 1)
        xA4_2_2 = torch.cat([x3_0A, xB_4], 1)
        xA4_2 = self.fu2(xA4_2_1, xA4_2_2)

        xA4_3_1 = torch.cat([self.pool2(x2_1B), self.pool2(xA_3)], 1)
        xA4_3_2 = torch.cat([x2_0A, xB_3], 1)
        xA4_3 = self.fu3(xA4_3_1, xA4_3_2)

        xA4_4_1 = torch.cat([self.pool2(x1_1B), self.pool2(xA_2)], 1)
        xA4_4_2 = torch.cat([x1_0A, xB_2], 1)
        xA4_4 = self.fu4(xA4_4_1, xA4_4_2)
        xA4_5_1 = torch.cat([self.pool2(x0_1B), self.pool2(xA_1)], 1)
        xA4_5_2 = torch.cat([x0_0A, xB_1], 1)
        xA4_5 = self.fu5(xA4_5_1, xA4_5_2)


        x4_1 = self.conv3_1(torch.cat([xA4_2, self.Up_6_1(xA4_0)], 1))

        x4_2 = self.conv3_2(torch.cat([xA4_3, self.Up_7_1(x4_1)], 1))

        x4_3 = self.conv3_3(torch.cat([xA4_4, self.Up_8_1(x4_2)], 1))

        x4_4 = self.conv3_4(torch.cat([xA4_5, self.Up_9_1(x4_3)], 1))


        xB4_0 = torch.cat([x4_1B, xA_5], 1)
        xB4_2 = torch.cat([x3_1B, xA_4], 1)
        xB4_3 = torch.cat([x2_1B, xA_3], 1)
        xB4_4 = torch.cat([x1_1B, xA_2], 1)
        xB4_5 = torch.cat([x0_1B, xA_1], 1)

        x1_9 = self.sff(xB4_2, self.Up_6(xB4_0), self.upsample_conv6(x4_1))
        # x1_9 = self.conv3_6(self.sff(xB4_2, self.Up_6(xB4_0), self.upsample_conv6(x4_1)))
        x1_10 = self.sff2(xB4_3, self.Up_7(x1_9), self.upsample_conv7(x4_2) )
        x1_11 = self.sff3(xB4_4, self.Up_8(x1_10), self.upsample_conv8(x4_3))
        x1_12 = self.sff4(xB4_5, self.Up_9(x1_11), self.upsample_conv9(x4_4))

        outo_o1 = self.conv_final(x4_4)
        outo_o2 = self.conv_final2(x1_12)

        # outo_o2 = self.conv_final2(x5_10)

        return outo_o2, outo_o1, xBB+xBB2, xAAA

