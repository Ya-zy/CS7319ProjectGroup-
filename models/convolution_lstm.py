import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import logging
import torch.functional as F
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


class Final_layer(nn.Module):

    def __init__(self,inplanes,planes,layers=[128,64,64]):
        super(Final_layer,self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(inplanes,layers[0],kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(layers[0],momentum=BN_MOMENTUM),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(layers[0], layers[1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(layers[1], momentum=BN_MOMENTUM),
        )
        self.downsample3 = None
        self.downsample4 = nn.Sequential(
            nn.Conv2d(layers[2], planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.block1 = Residual(inplanes,layers[0],downsample = self.downsample1)

        self.block2 = Residual(layers[0],layers[1],downsample=self.downsample2)

        self.block3 = Residual(layers[1],layers[2],downsample=self.downsample3)

        self.block4 = Residual(layers[2], planes, downsample=self.downsample4)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x


class element_conv(nn.Module):

    def __init__(self,inplanes,planes,midplanes,kernel,stride,padding):  # 128,128,64
        super(element_conv,self).__init__()
        if inplanes!=midplanes:
            self.downsample1 = nn.Sequential(
                nn.Conv2d(inplanes,midplanes,kernel_size=kernel,stride=stride,padding=padding,bias=False),
                nn.BatchNorm2d(midplanes,momentum=BN_MOMENTUM),
            )
        else:
            self.downsample1 = None
        self.downsample2 = None
        self.downsample3 = None
        if midplanes!=planes:
            self.downsample4 = nn.Sequential(
                nn.Conv2d(midplanes,planes,kernel_size=kernel,stride=stride,padding=padding,bias=False),
                nn.BatchNorm2d(planes,momentum=BN_MOMENTUM),
            )
        else:
            self.downsample4=None
        self.block1 = Residual(inplanes,midplanes,downsample = self.downsample1)
        self.block2 = Residual(midplanes, midplanes, downsample = self.downsample2)
        self.block3 = Residual(midplanes, midplanes, downsample = self.downsample3)
        self.block4 = Residual(midplanes, planes, downsample = self.downsample4)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x



class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        if input_channels!=hidden_channels:
            self.mid_channels = max(input_channels,hidden_channels)//2  # it will be used to add more layers
        else:
            self.mid_channels = input_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.ele_conv_x = element_conv(self.input_channels, self.hidden_channels,
                                       self.mid_channels,self.kernel_size,1,self.padding)
        self.ele_conv_h = element_conv(self.hidden_channels,self.hidden_channels,
                                       self.mid_channels,self.kernel_size,1,self.padding)

        self.Wxi = nn.Sequential(self.ele_conv_x,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=True))
        self.Whi = nn.Sequential(self.ele_conv_h,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=False))
        self.Wxf = nn.Sequential(self.ele_conv_x,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=True))
        self.Whf = nn.Sequential(self.ele_conv_h,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=False))
        self.Wxc = nn.Sequential(self.ele_conv_x,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=True))
        self.Whc = nn.Sequential(self.ele_conv_h,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=False))
        self.Wxo = nn.Sequential(self.ele_conv_x,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=True))
        self.Who = nn.Sequential(self.ele_conv_h,nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, 1, self.padding, bias=False))
        self.Bn = nn.BatchNorm2d(self.hidden_channels)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        # past cell
        self.p_c = None
        self.p_h = None

    def forward(self, x):  # [batch,12,64,48]
        if self.Wci is None:
            self.init_hidden(x.size())
        if self.p_c is None:
            self.init_state(x.size())

        ci = torch.sigmoid(self.Bn(self.Wxi(x) + self.Whi(self.p_h) + self.p_c * self.Wci))
        cf = torch.sigmoid(self.Bn(self.Wxf(x) + self.Whf(self.p_h) + self.p_c * self.Wcf))
        cc = cf * self.p_c + ci * torch.tanh(self.Bn(self.Wxc(x) + self.Whc(self.p_h)))
        co = torch.sigmoid(self.Bn(self.Wxo(x) + self.Who(self.p_h) + cc * self.Wco))
        ch = co * torch.tanh(cc)
        self.p_c = cc
        self.p_h = ch
        return ch
    def reset_state(self, pc = None, ph = None):
        # print("self.pc is{}".format(pc))
        self.p_c = pc
        self.p_h = ph
    def init_hidden(self, shape):  #  shape:[batch,keypoints,64,48]
        self.Wci = Variable(torch.zeros(shape[0], self.hidden_channels, shape[2], shape[3]).cuda()) # [batch,128,64,48]
        self.Wcf = Variable(torch.zeros(shape[0], self.hidden_channels, shape[2], shape[3]).cuda())
        self.Wco = Variable(torch.zeros(shape[0], self.hidden_channels, shape[2], shape[3]).cuda())
    def init_state(self, shape): #  shape:[batch,inf,keypoints,64,48]
        self.p_c = Variable(torch.zeros(shape[0],self.hidden_channels,shape[3],shape[4]).cuda())  # [batch,128,64,48]
        self.p_h = Variable(torch.zeros(shape[0],self.hidden_channels,shape[3],shape[4]).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, hidden_channels=[128,64,64], n=12, directory =None):
        super(ConvLSTM, self).__init__()
        self.e1 = ConvLSTMCell(n,hidden_channels[0],5)
        self.e2 = ConvLSTMCell(hidden_channels[0],hidden_channels[1],5)
        self.e3 = ConvLSTMCell(hidden_channels[1],hidden_channels[2],5)

        self.p1 = ConvLSTMCell(n, hidden_channels[0], 5)
        self.p2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], 5)
        self.p3 = ConvLSTMCell(hidden_channels[1], hidden_channels[2], 5)
        # self.final_layer = nn.Conv2d(sum(hidden_channels),n,1)
        self.final_layer = Final_layer(sum(hidden_channels),n)
        self.n = n
        self.directory = directory

    # the size of input should be [batch,inf,keypoints,64,48]
    def forward(self, input):
        inp_size = input.size()
        self.e1.init_state(inp_size)
        self.e2.init_state(inp_size)
        self.e3.init_state(inp_size)

        for i in range(inp_size[1]):
            xi = input[:,i,:,:,:] # [batch,12,64,48]
            h1 = self.e1(xi)  # (batch,128,64,48)
            h2 = self.e2(h1)  # (batch,64,64,48)
            self.e3(h2)  # (batch,64,64,48)

        self.p1.reset_state(self.e1.p_c, self.e1.p_h)
        self.p2.reset_state(self.e2.p_c, self.e2.p_h)
        self.p3.reset_state(self.e3.p_c, self.e3.p_h)

        # 只预测一张照片就循环一次
        size = input.size()
        h1 = self.p1(Variable(torch.zeros((inp_size[0], self.n, inp_size[3], inp_size[4]), dtype=torch.float32)).cuda())
        h2 = self.p2(h1)
        h3 = self.p3(h2)
        h = torch.cat([h1,h2,h3],1)  # (batch,256,64,48)
        ans = self.final_layer(h)  # (batch,12,64,48)
        return ans

    def init_weights(self,pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> pretrain the model')
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> pretrained distillator with the model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=True)

def get_convLSTM(cfg, is_train):
    model = ConvLSTM(cfg.MODEL.LSTM_H_CHANNELS,cfg.MODEL.LSTM_N)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        logger.info("start to init ConvLSTM weights")
        model.init_weights(cfg.MODEL.PRETRAINED_CONVLSTM)

    return model
