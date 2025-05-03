import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
            # nn.BatchNorm2d(high_dim)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
    # def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class fu1(nn.Module):
    def __init__(self, planes):
        super(fu1, self).__init__()

        self.ca = ChannelAttention(2*planes)
        self.sa = SpatialAttention()
        self.conv1x1 = conv1x1(2*planes, planes)
        self.w = nn.Parameter(torch.ones(2))
        self.z = nn.Parameter(torch.ones(2))


    def forward(self, x, x1):
        xc = torch.cat([x, x1], 1)
        xc = self.ca(xc) * xc
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        z1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        z2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        x = (self.sa(x)+w1*self.sa(x1)) * x
        x1 = (w2*self.sa(x)+self.sa(x1)) * x1

        x2 = torch.cat([x, x1], 1)+xc
        # x2 = x+x1

        return x2



def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """

    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B, h*w, K)

        context = torch.matmul(sim_map, value)  # (B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B, C, h, w)

        context = self.out_project(context)  # (B, C, h, w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]
class conv_block_nested3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested3, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)


    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)


        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel*4, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel*2)
        self.conv_big2 = conv_1x1(in_channel * 2, in_channel * 2)
        self.catconv = conv_block_nested3(in_channel * 5, in_channel,in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel*2,
            query_in_channels=in_channel*2,
            transform_channels=in_channel*2 // 2,
            out_channels=in_channel*2,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
        self.se = SE_Block(in_channel)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, x3):

        x1_1 = self.conv_big(x1)
        x4 = self.sigmoid(x1_1 + x2+x3)
        x2 = x2 *x4
        x3 = x3 *x4
        x = self.conv_small(torch.cat([x2,x3],1))

        x1 = self.se(torch.mul(x,x1))+x1

        out = self.catconv(torch.cat([x1,x2,x3], dim=1))
        return out





