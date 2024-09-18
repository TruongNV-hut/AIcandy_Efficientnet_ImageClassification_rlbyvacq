"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(EfficientNetBlock, self).__init__()
        self.stride = stride
        self.expand = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels * expansion)
        self.depthwise = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.se_reduce = nn.Conv2d(in_channels * expansion, in_channels * expansion // 4, kernel_size=1)
        self.se_expand = nn.Conv2d(in_channels * expansion // 4, in_channels * expansion, kernel_size=1)
        self.project = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = Swish()(self.bn0(self.expand(x)))
        out = Swish()(self.bn1(self.depthwise(out)))

        # Squeeze and Excitation
        se_out = torch.mean(out, (2, 3), keepdim=True)
        se_out = Swish()(self.se_reduce(se_out))
        se_out = torch.sigmoid(self.se_expand(se_out))
        out = out * se_out

        out = self.bn2(self.project(out))

        if self.stride == 1 and identity.size() == out.size():
            out = out + identity
        return out

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            EfficientNetBlock(32, 16, expansion=1, stride=1),
            EfficientNetBlock(16, 24, expansion=6, stride=2),
            EfficientNetBlock(24, 40, expansion=6, stride=2),
            EfficientNetBlock(40, 80, expansion=6, stride=2),
            EfficientNetBlock(80, 112, expansion=6, stride=1),
            EfficientNetBlock(112, 192, expansion=6, stride=2),
            EfficientNetBlock(192, 320, expansion=6, stride=1),
        )
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = Swish()(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = Swish()(self.bn2(self.conv_head(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.fc(x)
        return x
