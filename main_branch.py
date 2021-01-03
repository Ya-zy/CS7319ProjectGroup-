#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import argparse
from models import pose_resnet
from models.branch_net import LinearModel, weight_init
from data.penn_data import PennData
from data.penn_data_Test import PennDataTest
from src.utils import *
import sys
import warnings
import torch.backends.cudnn as cudnn
import time
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from src import cfg
from src import utils
import matplotlib.pyplot as plt
from pylab import *
import src.log as log
import src.checkflops as flops
import torchvision.models.resnet
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--config', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', type=str, default='./dataset/', help='path to dataset')
    parser.add_argument('--cuda', default=1, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')
    parser.add_argument('--load', type=str, default='', help='path to load a pretrained checkpoint')
    parser.add_argument('--resume', dest='resume', action='store_true', help='resume to train')
    parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
    parser.add_argument('--test', dest='test', action='store_true', help='test')
    parser.add_argument('--max_norm', dest='max_norm', action='store_true', help='maxnorm constraint to weights')
    parser.set_defaults(max_norm=True)
    args = parser.parse_args()
    return args


# # save hooks
# activation={}
def main(args):
    # logger file
    str_time = time.strftime('%Y-%m-%d')
    log_file = os.path.join(args.logDir, 'train_{}.log'.format(str_time))
    logging.basicConfig(filename=log_file, filemode='w', format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    warnings.filterwarnings('ignore')
    logging.info('>>> start')

    # config
    config = cfg.load_config(args.config)
    start_epoch = 0
    glob_step = 0
    pck_best =-1
    lr_now = config.TRAIN.LR
    lr_init = config.TRAIN.LR
    # hyper parameter
    data_dir = args.dataDir

    # create and init model
    logging.info(">>> creating  the backbone of model")
    backbone = pose_resnet.get_pose_net(config, is_train=True)
    branch_HR = LinearModel()
    branch_LR = LinearModel()

    # choice num of GPU
    device_ids = [0,1,2,3]
    if args.cuda:
        backbone = backbone.cuda(device_ids[0])
        backbone = nn.DataParallel(backbone, device_ids=device_ids)  # multi-Gpu
        branch_HR = branch_HR.cuda(device_ids[0])
        branch_HR = nn.DataParallel(branch_HR, device_ids=device_ids)
        branch_LR = branch_LR.cuda(device_ids[0])
        branch_LR = nn.DataParallel(branch_LR, device_ids=device_ids)

    # the backbone has already been pretrained
    # critrion and optimizer are all for distillator_net
    criterion_test = nn.MSELoss(size_average=True, reduction='sum').cuda()
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    optim_params = list(backbone.parameters()+branch_HR.parameters()+branch_LR.parameters())
    optimizer = optim.Adam(optim_params, lr=config.TRAIN.LR)

    # load ckpt
    if args.load:
        logging.info(">>> loading ckpt from '{}'".format(args.load))
        ckpt = torch.load(args.load)
        start_epoch = ckpt['epoch']
        pck_all = ckpt['pck']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        backbone.load_state_dict(ckpt['state_dict_backbone'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logging.info(">>> ckpt loaded (epoch: {} |pck:{})".format(start_epoch, pck_all))
    if args.resume:
        logger = log.Logger(os.path.join(args.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(args.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'pck'])

    # data loading
    logging.info(">>> loading data")

    if args.test:
        test_loader = DataLoader(dataset=PennDataTest(data_dir=data_dir, config=config),
                                 batch_size=config.TEST.BATCH_SIZE, shuffle=False, num_workers=config.TEST.JOBS,
                                 pin_memory=True)
        loss_test, pck_batch_02, pck_batch_01 = test(test_loader, backbone, criterion_test, config)
        logging.info(">>>>>> TEST results:")
        logging.info(">>>\nERRORS: {}".format(loss_test))
        sys.exit()

    # load dataset for training
    # Build dataset
    train_data = PennData(data_dir=data_dir, train=True, config=config)
    test_data = PennDataTest(data_dir=data_dir, config=config)
    logging.info('Train dataset total number' + str(len(train_data)))
    logging.info('Test dataset total number' + str(len(test_data)))

    # Data Loader
    train_loader = DataLoader(dataset=train_data, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=config.TRAIN.JOBS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=config.TEST.JOBS, pin_memory=True, drop_last=True)
    logging.info("dataset loaded!")

    # Increased operating efficiency
    cudnn.benchmark = True

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        logging.info('epoch:' + str(epoch))

        # per epoch
        glob_step, lr_now, loss_train = train(
            train_loader, backbone, branch_HR, branch_LR, criterion,criterion_test, optimizer, config.TRAIN.BATCH_SIZE,
            lr_init=lr_init, lr_now=lr_now, glob_step=glob_step, lr_decay=config.TRAIN.LR_DECAY,
            max_norm=args.max_norm)
        lr_init = lr_now
        loss_train = loss_train.item()
        loss_test, pck_num, label_num = test(test_loader, backbone,branch_HR, branch_LR, criterion_test, config)
        loss_test = loss_test.item()
        # update log file
        pck_epoch = pck_num / label_num
        logging.info('label_num:{}'.format(label_num))
        logging.info("pck:{}".format(pck_epoch))

        logger.append([epoch + 1, lr_now, loss_train, loss_test, pck_epoch],['int', 'float', 'float', 'float', 'float'])

        is_best = pck_epoch > pck_best
        pck_best = max(pck_epoch, pck_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'pck': pck_epoch,
                           'state_dict_backbone': backbone.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=args.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'pck': pck_epoch,
                           'state_dict_backbone': backbone.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=args.ckpt,
                          is_best=False)
    logger.close()


def train(train_loader, backbone, branch_HR, branch_LR, criterion,MSE_criterion, optimizer,batchsize,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, max_norm=True):
    backbone = backbone.train()
    branch_HR = branch_HR.train()
    branch_LR = 
    # backbone.start_flops_count()
    start = time.time()
    batch_time = 0
    loss = 1000
    check = True
    for step, (images, labels) in enumerate(train_loader):
        glob_step += 1
        print('lr_decay')
        print(glob_step%lr_decay)
        if glob_step % lr_decay == 0:
            lr_now = lrate_decay(optimizer, lr_init)
        with torch.no_grad():
            labels = Variable(labels.cuda()).long()  # [batch]
            images = Variable(images.cuda())  # [batch, 3, 112, 112]
            images_HR = images.view(-1, 3, 112, 112)
            images_LR = F.inertpolate(images_HR,scale_factor=0.5)
            images_LR = F.inertpolate(images_LR,size=[112,112],mode="bilinear")  # [batch, 3, 112, 112]
            images_all = torch.cat(images_HR,images_LR,0)   # [batch*2, 3, 112, 112]
        outputs, v_feature = backbone(images_all)  # [2*batch, classes]
        outputs_all, all_feature = branch(v_feature)
        outputs_x = outputs_all[0:batchsize]
        outputs_z = output_all[batchsize,2*batchsize]
        x_feature = all_feature[0:batchsize]    # [batch, 512]
        z_feature = all_feature[batchsize,2*batchsize]
        # the loss of x
        loss_x1 = criterion(outputs_x, labels)  # CrossEnctropyLoss
        loss_x2 = L_c(x_feature, labels)
        loss_x = loss_x1 + 0.0008*loss_x2
        # the loss of z
        loss_z1 = criterion(outputs_z, labels)  # CrossEnctropyLoss
        loss_z2 = L_c(z_feature, labels)
        loss_z = loss_z1 + 0.0008 * loss_z2
        loss_xz = MSE_criterion(x_feature,z_feature)

        loss = loss_x+loss_z+ 0.008*loss_xz
        optimizer.zero_grad()
        loss.backward()
        loss = loss.cpu().detach().numpy()
        # calculate loss
        # ******************** calculate and save loss of each joints ********************
        if step % 80 == 0:
            # logging.info(str(predict_heatmaps[0].shape)+"  "+str(label_map.shape)+"  "+str(len(imgs[0])))
            logging.info('step:{} loss1:{} loss2:{}'.format(str(step), str(float(loss1)), str(float(loss2))))
        # Gradient clipping
        if max_norm:
            optim_params = list(backbone.parameters()+branch.parameters())
            nn.utils.clip_grad_norm(optim_params, max_norm=100)
        optimizer.step()
    return glob_step, lr_now, loss


def test(test_loader, backbone,branch, criterion_test, config):
    backbone.eval()
    branch.eval()
    pck_amount = 0
    label_amount = 0
    with torch.no_grad():
        for step, (images, labels, test_set) in enumerate(test_loader):
            test_set = test_set.view(130, 3, 112, 112)
            images = images.view(1, 3, 112, 112)
            labels = labels.view(1, 3, 112, 112)
            all_set = torch.cat((test_set, images, labels), dim=0)  # [132,3,112,112]
            all_set = Variable(all_set.cuda())  # [132,3,112,112]
            outputs, v_features = backbone(all_set)  # [132, classes], [132, 512]
            output_b, x_features = branch(v_features)   # [132,classes], [132, 512]

            loss1 = criterion_test(x_features[130], x_features[131])
            loss_min = loss1
            for i in range(130):
                loss_all = criterion_test(v_features[130], v_features[i])
                if loss_all < loss1:
                    loss_min = loss_all
            loss = loss1
            batch_size = images.size(0)
            label_amount += batch_size
            # calculate loss and pck
            # _, predicted = torch.max(outputs, 1)    # [batch]
            # pck_step = 0
            # for i in range(batch_size):
            #     label_amount += batch_size
            #     if labels[i] == predicted[i]:
            #         pck_amount += 1
            #         pck_step += 1
            # pck=pck_step/batch_size
            loss = loss.cpu().numpy()
            loss1 = loss1.cpu().numpy()
            if loss_min == loss1:
                pck = 1
                pck_amount += 1
            else:
                pck = 0
            if step % 50 == 0:
                print(step)
                logging.info('step:{} loss:{} pck:{}'.format(str(step), str(float(loss)), str(pck)))
    return loss, pck_amount, label_amount


def lrate_decay(optimizer, lr):
    # lr = lr * gamma ** (step/decay_step)
    lr = lr * 0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

def L_c(feature, label):
    #输入feature map和label,返回聚类方差
    center = torch.zeros(label.size()[1], feature.size()[1]).cuda()
    label = torch.argmax(label, dim=0)
    print('this is label_size:{}'.format(label.size()))
    num_cnt = torch.zeros(label.size()[0])
    center_loss = 0
    for i in range(feature.size()[0]):
        center[label[i],:] += feature[i,:]
        num_cnt[label[i]] += 1
    for i in range(center.size()[0]):
        if num_cnt[i] >0 :
            center[i,:] = center[i,:] / num_cnt[i]
    for i in range(feature.size()[0]):
        center_loss += torch.sum((feature[i,:] - center[label[i],:])**2)

    return center_loss

if __name__ == "__main__":
    args = parse_args()
    print("args:{}".format(args))
    main(args)
