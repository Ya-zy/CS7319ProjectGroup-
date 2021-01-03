from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
import logging
import os
import torch.utils.data.distributed

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pool(out)
        return out


class DistillatorNet(nn.Module):

    def __init__(self):
        super(DistillatorNet,self).__init__()
        self.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64,128)
        self.block2 = BasicBlock(128,256)
        self.block3 = BasicBlock(256,512)

    def forward(self, x):
        x=self.relu1(self.bn1(self.conv1(x)))
        x=self.pool1(x)
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)

        return x

    def init_weights(self,pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> pretrain the model')
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> pretrained distillator with the model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=True)
        else:
            logger.info('=> init distillator weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_distillator(cfg, is_train):
    model = DistillatorNet()
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        logger.info("start to init distillator weights")
        model.init_weights(cfg.MODEL.PRETRAINED_DISTILLATOR)

    return model
