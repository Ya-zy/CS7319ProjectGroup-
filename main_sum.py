#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import argparse, pdb
from models import pose_resnet,convolution_lstm,lstm_predictor
from data.penn_data import PennData
from data.penn_data_Test import PennDataTest
from data.lfw_dataset import Dataset_LFW
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
import torch.nn.functional as F
import src.checkflops as flops
import torchvision.models.resnet


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


    # choice num of GPU
    device_ids = [0,1,2,3]
    if args.cuda:
        backbone = backbone.cuda(device_ids[0])
        backbone = nn.DataParallel(backbone, device_ids=device_ids)  # multi-Gpu

    # the backbone has already been pretrained
    # critrion and optimizer are all for distillator_net
    criterion_test = nn.MSELoss(reduction='mean').cuda()
    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    optim_params = list(backbone.parameters())
    optimizer = optim.Adam(optim_params, lr=config.TRAIN.LR)

    # load ckpt
    if args.load:
        logging.info(">>> loading ckpt from '{}'".format(args.load))
        ckpt = torch.load(args.load)
        start_epoch = ckpt['epoch']
        pck_all = ckpt['pck']
        glob_step = ckpt['step']
        lr_now = 0.001
        backbone.load_state_dict(ckpt['state_dict_backbone'])
        optimizer.load_state_dict(ckpt['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        logging.info(">>> ckpt loaded (epoch: {} |pck:{})".format(start_epoch, pck_all))
    if args.resume:
        logger = log.Logger(os.path.join(args.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(args.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'pck'])

    # data loading
    logging.info(">>> loading data")

    if args.test:
        # test_loader = DataLoader(dataset=PennDataTest(data_dir=data_dir, config=config),
        #                          batch_size=config.TEST.BATCH_SIZE, shuffle=False, num_workers=config.TEST.JOBS,
        #                          pin_memory=True)
        # loss_test, pck_num, label_num = test(test_loader, backbone, criterion_test, config)
        # logging.info(">>>>>> TEST results:")
        # logging.info(">>>\nERRORS: {}, pck_num:{}, label_num:{}".format(loss_test, pck_num, label_num))
        # sys.exit()
        test_loader = DataLoader(dataset=Dataset_LFW(root='dataset/LFW/LFW_120x120_120',
                                 data_list_file='dataset/LFW/lfw_test_pair.txt'),
                                 batch_size=100, shuffle=False, num_workers=config.TEST.JOBS,
                                 pin_memory=True)
        accuracy, thd = eval(backbone,test_loader,100)
        logging.info(">>>>>> TEST results:")
        logging.info(">>>\naccuracy:{}, thd:{}".format(accuracy, thd))



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
    centers = torch.randn(10575, 512) # 在这加上了centers
    centers = Variable(centers.cuda())

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        logging.info('epoch:' + str(epoch))

        # per epoch, train的输入中也加入了centers
        glob_step, lr_now, loss_train = train(
            train_loader, backbone, criterion, optimizer, config.TRAIN.BATCH_SIZE, centers,
            lr_init=lr_init, lr_now=lr_now, glob_step=glob_step, lr_decay=config.TRAIN.LR_DECAY,
            max_norm=args.max_norm)
        lr_init = lr_now
        loss_train = loss_train.item()
        loss_test, pck_num, label_num = test(test_loader, backbone, criterion_test, config)
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


def train(train_loader, backbone,criterion, optimizer,batchsize, centers,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, max_norm=True):
    backbone = flops.add_flops_counting_methods(backbone)
    backbone = backbone.train()
    # backbone.start_flops_count()
    start = time.time()
    batch_time = 0
    loss = 1000
    check = True
    for step, (images, labels) in enumerate(train_loader):
        glob_step += 1
        #print('lr_decay')
        #print(glob_step%lr_decay)
        if glob_step % lr_decay == 0:
            lr_now = lrate_decay(optimizer, lr_init)
        with torch.no_grad():
            labels = Variable(labels.cuda()).long()  # [batch]
            images = Variable(images.cuda())  # [batch, 3, 112, 112]
            images = images.view(-1, 3, 112, 112)
        outputs, v_feature = backbone(images)  # [batch, classes]
        # print('Flops')
        # print(backbone.compute_average_flops_cost())
        loss1 = criterion(outputs, labels)  # CrossEnctropyLoss
        #one_hot_label= nn.functional.one_hot(labels,10575)    # [batch, 10575]
        #print('the size of one_hot_label:{}'.format(one_hot_label.size()))
        loss2 = L_c(v_feature, labels)
        #loss2, centers = L_c(v_feature, labels, centers)
        loss = loss1 + 0.008*loss2
        optimizer.zero_grad()
        loss.backward()
        loss = loss.cpu().detach().numpy()
        # calculate loss
        # ******************** calculate and save loss of each joints ********************
        if step % 80 == 0:
            # logging.info(str(predict_heatmaps[0].shape)+"  "+str(label_map.shape)+"  "+str(len(imgs[0])))
            logging.info('step:{} loss1:{} loss2:{}'.format(str(step), str(float(loss1)), str(float(0.0004*loss2))))
        # Gradient clipping
        if max_norm:
            optim_params = list(backbone.parameters())
            nn.utils.clip_grad_norm(optim_params, max_norm=100)
        optimizer.step()
    return glob_step, lr_now, loss


def test(test_loader, backbone, criterion_test, config):
    backbone.eval()
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

            loss1 = criterion_test(v_features[130], v_features[131])
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
            loss_min = loss_min.cpu().numpy()
            if loss_min == loss1:
                pck = 1
                pck_amount += 1
            else:
                pck = 0
            if step % 50 == 0:
                #deltaloss = loss1 - loss_min
                print(step)
                #logging.info('{}'.format(str(float(deltaloss))))
                logging.info('step:{} loss:{} pck:{}'.format(str(step), str(float(loss)), str(pck)))
    return loss, pck_amount, label_amount

def eval(model, loader, test_batch_size=100):
    model.eval()
    # eval_dataset = Dataset_LFW(test_root, test_list)
    # loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, batch_size=test_batch_size)
    predicts = []
    for ite, (batch_img1, batch_img2, batch_path_1, batch_path_2, is_same) in enumerate(loader):
        img1, img2 = batch_img1.numpy(), batch_img2.numpy()
        imglist = [img1, img1, img2, img2]
        img = np.vstack(imglist)
        img = torch.from_numpy(img).float().cuda()

        out, features = model(img)
        print('this is the shape of out:{}'.format(out.shape))

        for b in range(test_batch_size):
            f1, f2 = features[b], features[b+2*test_batch_size]
            # print('f1.shape=', f1.shape, 'f2.shape=', f2.shape)
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

            temp_is_same = is_same[b]
            path_1 = batch_path_1[b].split('/')
            path_2 = batch_path_2[b].split('/')
            temp_path_1 = path_1[-2] + path_1[-1]
            temp_path_2 = path_2[-2] + path_1[-1]
            predicts.append('{}\t{}\t{}\t{}\n'.format(temp_path_1, temp_path_2, cosdistance, temp_is_same))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts_list = []

    print('len predicts=', len(predicts))
    for line in predicts:
        line = line.strip('\n').split('\t')
        predicts_list.append(line)

    predicts = np.array(predicts_list)

    for idx, (train_index, test_index) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train_index])
        accuracy.append(eval_acc(best_thresh, predicts[test_index]))
        thd.append(best_thresh)
    print('\nLFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    print('acc_list:', np.round(np.asarray(accuracy), 4))
    return np.mean(accuracy), np.mean(thd)

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
    center = torch.zeros(10575, feature.size()[1])
    center = Variable(center).cuda()
    #print(center.size())
    #label = torch.argmax(label, dim=0)
    #print('this is label_size:{}'.format(label.size()))
    num_cnt = torch.zeros(10575)
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
'''
def L_c(feature, label, center, momentum=0.9):
    #输入feature map和label,返回聚类方差
    delta_c = torch.zeros(10575, feature.size()[1])
    delta_c = Variable(delta_c).cuda()
    num_cnt = torch.zeros(10575)
    center_loss = 0
    for i in range(feature.size()[0]):
        delta_c[label[i],:] += feature[i,:]
        num_cnt[label[i]] += 1
    for i in range(delta_c.size()[0]):
        if num_cnt[i] >0 :
            center[i,:] = momentum * center[i,:] + (1-momentum) * delta_c[i,:] / num_cnt[i]
    for i in range(feature.size()[0]):
        center_loss += torch.sum((feature[i,:] - center[label[i],:])**2)

    return center_loss, center
'''
def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


if __name__ == "__main__":
    args = parse_args()
    print("args:{}".format(args))
    main(args)
