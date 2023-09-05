import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import TripletAttention
import warnings
warnings.filterwarnings("ignore")

class Residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, attention=True):
        super(Residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_ch)

        self.conv_branch = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.in_branch = nn.InstanceNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.attention = attention
        self.attention_module = Attention(out_ch, mode=3)

    def forward(self, x):

        x_identity = x
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.in2(x)

        if self.in_ch != self.out_ch:
            x_identity = self.conv_branch(x_identity)
            x_identity = self.in_branch(x_identity)

        out = self.relu(x_identity + x)
        if self.attention:
            out = self.attention_module(out)
        return out

class MyUNet_plus(nn.Module):
    def __init__(self, in_ch, act=True):
        super(MyUNet_plus, self).__init__()

        self.resblock1 = Residual_block(5, 2 * in_ch)
        self.resblock2 = Residual_block(2 * in_ch, 4 * in_ch)
        self.resblock2_2 = Residual_block(4 * in_ch, 4 * in_ch)
        self.resblock3 = Residual_block(4 * in_ch, 8 * in_ch)
        self.resblock3_2 = Residual_block(8 * in_ch, 8 * in_ch)

        self.resblock4 = Residual_block(8 * in_ch, 16 * in_ch)
        self.resblock5 = Residual_block(16 * in_ch, 8 * in_ch)
        self.resblock5_2 = Residual_block(8 * in_ch, 8 * in_ch)
        self.resblock6 = Residual_block(8 * in_ch, 4 * in_ch)
        self.resblock6_2 = Residual_block(4 * in_ch, 4 * in_ch)
        self.resblock7 = Residual_block(4 * in_ch, 2 * in_ch)
        self.lastconv = nn.Conv2d(2 * in_ch, 1, 1, padding=0)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16 * in_ch, 8 * in_ch, 1, padding=0)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8 * in_ch, 4 * in_ch, 1, padding=0)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(4 * in_ch, 2 * in_ch, 1, padding=0)
        )
        self.tanh = nn.Tanh()
        self.act = act

    def forward(self, x):
        x1 = self.resblock1(x)
        p1 = self.pool1(x1)

        x2 = self.resblock2(p1)
        x2 = self.resblock2_2(x2)
        p2 = self.pool2(x2)

        x3 = self.resblock3(p2)
        x3 = self.resblock3_2(x3)
        p3 = self.pool3(x3)

        x4 = self.resblock4(p3)
        u1 = self.up1(x4)
        merge1 = torch.cat([u1, x3], dim=1)

        x5 = self.resblock5(merge1)
        x5 = self.resblock5_2(x5)
        u2 = self.up2(x5)
        merge2 = torch.cat([u2, x2], dim=1)

        x6 = self.resblock6(merge2)
        x6 = self.resblock6_2(x6)
        u3 = self.up3(x6)
        merge3 = torch.cat([u3, x1], dim=1)

        x7 = self.resblock7(merge3)
        out = self.lastconv(x7)

        if self.act:
            out = self.tanh(out)
        return out

class MyUNet(nn.Module):
    def __init__(self, in_ch):
        super(MyUNet, self).__init__()

        self.resblock1 = Residual_block(1, 2 * in_ch)
        self.resblock2 = Residual_block(2 * in_ch, 4 * in_ch)
        self.resblock3 = Residual_block(4 * in_ch, 8 * in_ch)

        self.resblock4 = Residual_block(8 * in_ch, 16 * in_ch)
        self.resblock5 = Residual_block(16 * in_ch, 8 * in_ch)
        self.resblock6 = Residual_block(8 * in_ch, 4 * in_ch)
        self.resblock7 = Residual_block(4 * in_ch, 2 * in_ch)
        self.lastconv = nn.Conv2d(2 * in_ch, 1, 1, padding=0)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16 * in_ch, 8 * in_ch, 1, padding=0)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8 * in_ch, 4 * in_ch, 1, padding=0)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(4 * in_ch, 2 * in_ch, 1, padding=0)
        )
        self.tanh = nn.Tanh()


    def forward(self, x):
        x1 = self.resblock1(x)
        p1 = self.pool1(x1)

        x2 = self.resblock2(p1)
        p2 = self.pool2(x2)

        x3 = self.resblock3(p2)
        p3 = self.pool3(x3)

        x4 = self.resblock4(p3)
        u1 = self.up1(x4)
        merge1 = torch.cat([u1, x3], dim=1)

        x5 = self.resblock5(merge1)
        u2 = self.up2(x5)
        merge2 = torch.cat([u2, x2], dim=1)

        x6 = self.resblock6(merge2)
        u3 = self.up3(x6)
        merge3 = torch.cat([u3, x1], dim=1)

        x7 = self.resblock7(merge3)

        out = self.lastconv(x7)
        out = self.tanh(out)
        return out

# channel attention
class CAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x

# spatial attention
class SAM(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(SAM, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self.sigmoid = nn.Sigmoid()
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)
        x4 = self.conv4(F.relu_(x2 + x3))
        out = self.conv5(x4)
        return self.sigmoid(out)*x

class Attention(nn.Module):
    def __init__(self, in_channels, mode=1):
        super(Attention, self).__init__()

        self.cam = CAM(in_channels)
        self.sam = SAM(in_channels)
        self.triplet = TripletAttention()
        self.mode = mode

    def forward(self, x):
        if  self.mode == 1:
            return self.cam(x) + x
        elif self.mode == 2:
            return self.sam(x) + x
        elif self.mode == 3:
            return self.triplet(x)
        else:
            return self.cam(x) + self.sam(x) + x



if __name__ == '__main__':

    x = torch.FloatTensor(1, 5, 288, 288)
    net_plus = MyUNet_plus(32)
    net = MyUNet(32)

    out1 = net_plus(x)
    out2 = net(out1)
    print('out:', out1.shape, out2.shape)

