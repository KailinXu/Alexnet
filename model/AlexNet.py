# -*- coding: utf-8 -*-
'''
    Created on wed Sept 22 16:46 2018

    Author           : Kailin Xu
    Email            : 10405501@qq.com
    Last edit date   : Nov 9 9:48 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, chan_in, chan_out, stride=1, shortcut=None):
        '''
            Args:
                 chan_in       : (int) in channels
                 chan_out      : (int) out channels
                 stride        : (int) convolution stride
                 shortcut      : (nn.Module) shortcut module
        '''
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(chan_in ,chan_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_out, chan_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chan_out)
        )

        self.right = shortcut

    def forward(self, x):
        '''
            Args:
                 x              : (torch.FloatTensor\torch.cuda.FloatTensor) input tensor
        '''
        x_ = self.left(x)
        residual = x if self.right is None else self.right(x)
        x_ += residual

        return x_



# The definition of a AlexNet like network
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
