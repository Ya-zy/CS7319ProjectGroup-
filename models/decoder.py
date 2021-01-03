from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
import logging
import os
import torch.utils.data.distributed
import numpy as np

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class DecoderNet(nn.Module):
    def __init__(self,cfg):
        super(DecoderNet, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.inplanes = 512
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self,model, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            pretrained_state_dict = torch.load(pretrained)
            pretrained_decoder_state_dict = {}
            model_state_dict = model.state_dict()
            # in this part we will pretrain decoder with part of ResNet
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    pretrained_decoder_state_dict[k] = v
                # for 'final_layer.weight' and 'final_layer.bias', we need to change the size(0)from 17 to 12
                index = np.array([7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]) - 1
                final_weight_hat = torch.zeros(12, 256, 1, 1).cuda()
                final_bias_hat = torch.zeros(12).cuda()
                for i in range(12):
                    final_weight_hat[i] = pretrained_state_dict['final_layer.weight'][index[i]]
                    final_bias_hat[i] = pretrained_state_dict['final_layer.bias'][index[i]]
                pretrained_decoder_state_dict['final_layer.weight'] = final_weight_hat
                pretrained_decoder_state_dict['final_layer.bias'] = final_bias_hat

            # ******************************************************************
            model.load_state_dict(pretrained_decoder_state_dict,strict=True)
            logger.info('=> pretrained decoder with the model {}'.format(pretrained))
        else:
            logger.info('=> init decoder weights from normal distribution')
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


def get_decoder_net(cfg,is_train):
    model = DecoderNet(cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        logger.info("start to init decoder weights")
        model.init_weights(model, cfg.MODEL.PRETRAINED_RESNET34)

    return model