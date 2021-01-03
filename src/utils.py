import json
import numpy as np
import os
import scipy.misc
import src.log as log
import pickle
import torch.nn as nn
import torch

# init a dict to save loss of each temporal and the total loss
def loss_history_init(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history


def cal_loss(temporal, predict_heatmaps, label_map, criterion):

    loss_save = loss_history_init(temporal=temporal)  # a dict to save loss

    predict = predict_heatmaps[0]  # Tensor:size([batch,14,45,45])
    # label_map  Tensor, size:([batch, 5, 14, 45, 45])
    target = label_map[:, 0, :, :, :]  # Tensor:size([batch,14,45,45])
    initial_loss = criterion(predict, target)  # loss initial of the first frame for Convnet1
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t + 1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)  # loss in each stage

        total_loss += tmp_loss
        loss_save['temporal' + str(t)] = float('%.8f' % tmp_loss)
    total_loss = total_loss
    loss_save['total'] = float(total_loss)
    return loss_save, total_loss


def save_loss(predict_heatmaps, label_map, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save, total_loss = cal_loss(temporal, predict_heatmaps, label_map, criterion)
    # save loss to file
    if train is True:
        # if not os.path.exists(save_dir + 'loss_epoch' + str(epoch)):
        #     os.mkdir(save_dir + 'loss_epoch' + str(epoch))
        # json.dump(loss_save, open(save_dir + 'loss_epoch' + str(epoch) + '/s' + str(step).zfill(4) + '.json', 'wb'))
        # 关于是否isbest还没有修改
        log.save_ckpt(loss_save,ckpt_path=save_dir)
    else:
        if not os.path.exists(save_dir + 'loss_test/'):
            os.mkdir(save_dir + 'loss_test/')
        json.dump(loss_save, open(save_dir + 'loss_test/' + str(step).zfill(4) + '.json', 'wb'))

    return total_loss


def save_heatmap(label_map, predict_heatmaps, step, epoch, imgs, train, pck=1, save_dir='output/'):
    output_dir = '{}{}-{}-{}{}'.format(save_dir, train, str(epoch), str(step), '.pkl')
    data_dic =[]
    for i in range(label_map.shape[0]):
        data_dic.append([[j[i] for j in imgs], label_map[i], [j[i] for j in predict_heatmaps]])
    with open(output_dir, 'wb') as f:
        pickle.dump(data_dic, f) 


def save_images(label_map, predict_heatmaps, step, epoch, imgs, train, pck=1, temporal=5, save_dir='ckpt/'):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * joints *   45 * 45
    :param step:
    :param temporal:
    :param epoch:
    :param train:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """
    # 每100个step存一次 training heat map
    for b in range(label_map.shape[0]):                     # for each batch (person)
        output = np.ones((50 * 2, 50 * temporal))           # cd .. temporal save a single image
        seq = imgs[0][b].split('/')[-2]                     # sequence name 001L0
        img = ""
        for t in range(temporal):                           # for each temporal
            im = imgs[t][b].split('/')[-1][1:5]             # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))
            for i in range(21):                             # for each joint
                pre += np.asarray(predict_heatmaps[t][b, i, :, :].data)  # 2D
                gth += np.asarray(label_map[b, t, i, :, :].data)         # 2D

            output[0:45,  50 * t: 50 * t + 45] = gth
            output[50:95, 50 * t: 50 * t + 45] = pre

        if train is True:
            if not os.path.exists(save_dir + 'epoch'+str(epoch)):
                os.mkdir(save_dir + 'epoch'+str(epoch))
            scipy.misc.imsave(save_dir + 'epoch'+str(epoch) + '/s'+str(step) + '_b' + str(b) + '_' + seq + img + '.jpg', output)
        else:

            if not os.path.exists(save_dir + 'test'):
                os.mkdir(save_dir + 'test')
            scipy.misc.imsave(save_dir + 'test' + '/s' + str(step) + '_b' + str(b) + '_'
                              + seq + img + '_' + str(round(pck, 4)) + '.jpg', output)


def loss_evaluation(label_map, temporal_heatmaps, heatmap_multiplier,label,keypoints,seq, sigma=0.2):
    # labelmap:[batch,12,64,48], heatmap:[batch,12,64,48], label:[batch,3(xyv),14,seq]
    label_3 = label[:,0:3,1:13,seq-1].permute(0,2,1) # [batch,12,3]
    pck_eval_02 = 0
    zero_label_eval_02 =0
    pck_eval_01 = 0
    pck_eval_torso = 0
    pck_keypoints_02 = torch.zeros(keypoints,1)  # [12,1]
    pck_keypoints_01 = torch.zeros(keypoints,1) # [12,1]
    pck_keypoints_torso = torch.zeros(keypoints,1) # [12,1]
    zero_keypoints = torch.zeros(keypoints,1) # [12,1]
    label_keypoints = torch.zeros(keypoints,1) # [12,1]
    # we can not directly use the label there because the label is in the coordinate of origin
    coordinates_tar, _ = integrate_tensor_2d(label_map * heatmap_multiplier)  # The horizontal axis is x
    # the size of coordinate_pre is [batch,12,2]
    coordinates_temp, _ = integrate_tensor_2d(temporal_heatmaps * heatmap_multiplier)
    coordinates_tar = coordinates_tar.cpu().numpy()
    coordinates_temp = coordinates_temp.cpu().numpy()
    length = np.amax([label_map.shape[2],label_map.shape[3]])
    for b in range(label_map.shape[0]):        # size of label_map: batch_size*12*64*48
        pck02 = 0
        pck01 = 0
        pck_torso = 0
        zero_label = 0
        torso_length = 25
        torso_size = []
        coordinates_tar_person = coordinates_tar[b, :, :]  # [12,2]
        coordinates_pre_person = coordinates_temp[b,:,:]

        for k in range (coordinates_tar.shape[1]):
            if label[b,2,k+1,0]<1 or label[b,2,k+1,1]<1 or label[b,2,k+1,2]<1:
                zero_label += 1
                zero_keypoints[k] += 1
                continue
            x_pre,y_pre = coordinates_pre_person[k][0], coordinates_pre_person[k][1]
            x_tar,y_tar = coordinates_tar_person[k][0], coordinates_tar_person[k][1]

            dis = np.sqrt((x_pre - x_tar) ** 2 + (y_pre - y_tar) ** 2)
            if dis < sigma * length:
                pck02 += 1
                pck_keypoints_02[k] += 1
            if dis < 0.1 *length:
                pck01 += 1
                pck_keypoints_01[k] += 1
            if dis <sigma * torso_length:
                pck_torso+=1
                pck_keypoints_torso[k] += 1
        pck_eval_02 = pck_eval_02 + pck02
        pck_eval_01 = pck_eval_01 + pck01
        pck_eval_torso = pck_eval_torso +pck_torso
        zero_label_eval_02 += zero_label
        label_keypoints = label_map.shape[0]-zero_keypoints
    return pck_eval_02,pck_eval_01,pck_eval_torso,pck_keypoints_02,pck_keypoints_01,pck_keypoints_torso,\
           float(label_map.shape[0] * label_map.shape[1]-zero_label_eval_02),label_keypoints
# def loss_evaluation(label_map, temporal_heatmaps, heatmap_multiplier,label, sigma=0.04):
#     # labelmap:[batch,12,64,48], heatmap:[batch,12,64,48], label:[batch,3(xyv),14,seq]
#     pck_eval = 0
#     zero_label_eval =0
#     coordinates_tar, _ = integrate_tensor_2d(label_map * heatmap_multiplier)  # The horizontal axis is x
#     # coordinate_pre的size为 [batch,12,2]
#     coordinates_temp, _ = integrate_tensor_2d(temporal_heatmaps * heatmap_multiplier)
#     coordinates_tar = coordinates_tar.cpu().numpy()
#     coordinates_temp = coordinates_temp.cpu().numpy()
#     length = np.amax([label_map.shape[2],label_map.shape[3]])
#     for b in range(label_map.shape[0]):        # size of label_map: batch_size*12*64*48
#         pck = 0
#         zero_label = 0
#         coordinates_tar_person = coordinates_tar[b, :, :]
#         coordinates_pre_person = coordinates_temp[b,:,:]
#         for k in range (coordinates_tar.shape[1]):
#             if label[b,2,k+1,0]<1 or label[b,2,k+1,1]<1 or label[b,2,k+1,2]<1:
#                 zero_label += 1
#                 continue
#             x_pre,y_pre = coordinates_pre_person[k][0], coordinates_pre_person[k][1]
#             x_tar,y_tar = coordinates_tar_person[k][0], coordinates_tar_person[k][1]
#
#             dis = np.sqrt((x_pre - x_tar) ** 2 + (y_pre - y_tar) ** 2)
#             if dis < sigma * length:
#                 pck += 1
#         pck_eval = pck_eval + pck
#         zero_label_eval += zero_label
#     return pck_eval,float(label_map.shape[0] * label_map.shape[1]-zero_label_eval)


def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       12 * 64 * 48
    :param target:          3D numpy       12 * 64 * 48
    :param label_size:
    :param sigma:
    :return: 0/14, 1/14, ...
    """
    pck = 0
    for i in range(predict.shape[0]):
        # coordinate of max_value in predict_map
        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        # coordinate of max_value in target_map
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1
    return pck / float(predict.shape[0])


def integrate_tensor_2d(heatmaps, softmax=True, eps=1e-8):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / (mass_x.sum(dim=2, keepdim=True) + eps)
        y = y / (mass_y.sum(dim=2, keepdim=True) + eps)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps


def draw_loss(epoch):
    all_losses = os.listdir('ckpt/loss_epoch'+str(epoch))
    losses = []

    for loss_j in all_losses:
        loss = json.load('ckpt/loss_epoch'+str(epoch) + '/' +loss_j)
        a = loss['total']
        losses.append(a)


def Tests_save_label_imgs(label_map, predict_heatmaps, step, imgs, temporal=13, save_dir='ckpt/'):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * joints *   45 * 45
    :param step:
    :param temporal:
    :param epoch:
    :param train:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """

    for b in range(label_map.shape[0]):  # for each batch (person)
        output = np.ones((50 * 2, 50 * temporal))  # cd .. temporal save a single image
        seq = imgs[0][b].split('/')[-2]  # sequence name 001L0
        img = ""  # all image name in the same seq
        label_dict = {}  # all image label in the same seq
        pck_dict = {}
        for t in range(temporal):  # for each temporal
            labels_list = []  # 21 points label for one image [[], [], [], .. ,[]]

            im = imgs[t][b].split('/')[-1][1:5]  # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))

            # ****************** get pck of one image ************************
            target = np.asarray(label_map[b, t, :, :, :].data)  # 3D numpy 21 * 45 * 45
            predict = np.asarray(predict_heatmaps[t][b, :, :, :].data)  # 3D numpy 21 * 45 * 45
            empty = np.zeros((21, 45, 45))

            if not np.equal(empty, target).all():
                pck = PCK(predict, target, sigma=0.04)
                pck_dict[seq + '_' + im] = pck

            # ****************** save image and label of 21 joints ******************
            for i in range(21):  # for each joint
                gth += np.asarray(label_map[b, t, i, :, :].data)  # 2D
                tmp_pre = np.asarray(predict_heatmaps[t][b, i, :, :].data)  # 2D
                pre += tmp_pre

                #  get label of original image
                corr = np.where(tmp_pre == np.max(tmp_pre))
                x = corr[0][0] * (256.0 / 45.0)
                x = int(x)
                y = corr[1][0] * (256.0 / 45.0)
                y = int(y)
                labels_list.append([y, x])  # save img label

            output[0:45, 50 * t: 50 * t + 45] = gth  # save image
            output[50:95, 50 * t: 50 * t + 45] = pre

            label_dict[im] = labels_list  # save label

        # calculate average PCK
        # print pck_dict
        avg_pck = sum(pck_dict.values()) / float(pck_dict.__len__())
        print('step ...%d ... PCK %f  ....' % (step, avg_pck))

        # ****************** save image ******************
        if not os.path.exists(save_dir + 'test'):
            os.mkdir(save_dir + 'test')
        scipy.misc.imsave(save_dir + 'test' + '/s' + str(step) + '_'
                          + seq + img + '_' + str(round(avg_pck, 4)) + '.jpg', output)

        # ****************** save label ******************
        if not os.path.exists(os.path.join(save_dir, 'test_predict')):
            os.mkdir(os.path.join(save_dir, 'test_predict'))

        save_dir_label = os.path.join(save_dir, 'test_predict') + '/' + seq
        if not os.path.exists(save_dir_label):
            os.mkdir(save_dir_label)

        json.dump(label_dict, open(save_dir_label + '/' + str(step) + '.json', 'w'), sort_keys=True, indent=4)
        return pck_dict



from PIL import Image
from PIL import ImageDraw



def draw_point(points, im):
    """
    draw key point on image
    :param points: list 21 [ [x1,y1], ..., [x21,y21]  ]
    :param im: PIL Image
    :return:
    """
    i = 0
    draw=ImageDraw.Draw(im)

    for point in points:
        x = point[1]
        y = point[0]

        if i==0:
            rootx=x
            rooty=y
        if i==1 or i==5 or i==9 or i==13 or i==17:
            prex=rootx
            prey=rooty

        if i >0 and i<=4:
            draw.line((prex,prey,x,y),'red')
            draw.ellipse((x-3, y-3, x+3, y+3), 'red', 'black')
        if i >4 and i<=8:
            draw.line((prex,prey,x,y),'yellow')
            draw.ellipse((x-3, y-3, x+3, y+3), 'yellow', 'black')

        if i >8 and i<=12:
            draw.line((prex,prey,x,y),'green')
            draw.ellipse((x-3, y-3, x+3, y+3), 'green', 'black')
        if i >12 and i<=16:
            draw.line((prex,prey,x,y),'blue')
            draw.ellipse((x-3, y-3, x+3, y+3), 'blue', 'black')
        if i >16 and i<=20:
            draw.line((prex,prey,x,y),'purple')
            draw.ellipse((x-3, y-3, x+3, y+3), 'purple', 'black')


        prex=x
        prey=y
        i=i+1
    return im


