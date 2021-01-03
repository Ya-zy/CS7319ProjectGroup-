import torch.nn as nn
import torch.nn.functional as F
import torch
from models import convolution_lstm
import logging
from pylab import *

import torch.utils.data.distributed
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Residual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class _make_net2(nn.Module):
    def __init__(self,inplanes,planes,layers = [128,128,64,64]):
        super(_make_net2,self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(inplanes, layers[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(layers[0], momentum=BN_MOMENTUM),
        )
        self.downsample2 = None

        self.downsample3 = nn.Sequential(
            nn.Conv2d(layers[1], layers[2], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(layers[2], momentum=BN_MOMENTUM),
        )
        self.downsample4 = None

        self.downsample5 = nn.Sequential(
            nn.Conv2d(layers[3], planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.block1 = Residual(inplanes, layers[0], downsample=self.downsample1)

        self.block2 = Residual(layers[0], layers[1], downsample=self.downsample2)

        self.block3 = Residual(layers[1], layers[2], downsample=self.downsample3)

        self.block4 = Residual(layers[2], layers[3], downsample=self.downsample4)

        self.block5 = Residual(layers[3], planes, downsample=self.downsample5)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class _make_net3(nn.Module):

    def __init__(self, inplanes, planes):
        super(_make_net3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,128,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,planes,kernel_size=1,padding=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


class LSTMPredictor(nn.Module):
    def __init__(self, config,istrain):
        super(LSTMPredictor, self).__init__()
        self.keypoints = config.MODEL.NUM_JOINTS
        self.convnet1 = convolution_lstm.get_convLSTM(config,is_train=istrain)
        self.convnet2 = _make_net2(24,12)

    def forward(self, heatmap12,spatial_heatmap3,batchsize,step):
        # the size of heatmap12 :[batch,seq-1,keypoints,64,48]
        # the size of heatmap3 :[batch,keypoints,64,48]
        # the size of label_map :[batch,keypoints,64,48]
        temp_heatmap3 = self.convnet1(heatmap12)  # [batch,keypoints,64,48]
        heatmap3 = torch.cat([temp_heatmap3, spatial_heatmap3], dim=1)  # [batch,2*keypoints,64,48]
        if step == 1000:
            for i in range(12):
                figure()
                imshow(temp_heatmap3[0,i,:,:].detach().cpu().numpy()) # size of temp [64,48]
        predict_heatmap3 = self.convnet2(heatmap3)  # [batch,keypoints,64,48]

        return predict_heatmap3,temp_heatmap3