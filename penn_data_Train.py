'''
only used for penn_action datasets
'''
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import pickle


class PennDataTest(Dataset):
    def __init__(self, config, data_dir='dataset/'):

        self.input_h = config.MODEL.INPUT_H  # 112
        self.input_w = config.MODEL.INPUT_W  # 112
        self.data_dir = data_dir + 'test/'  # dataset/test/
        self.subdir = ['mugshot_frontal_cropped_all', 'surveillance_cameras_distance_1', 'surveillance_cameras_distance_2', 'surveillance_cameras_distance_3']
        self.label_dir = self.subdir[0]  # ['mugshot_frontal_cropped_all']
        self.label_path = os.path.join(self.data_dir,self.subdir[0])
        self.labels = os.listdir(self.label_path)
        self.labels.sort()
        self.images_path = []
        for i in range(3):
            for j in range(5):
                image_list = os.listdir(os.path.join(self.data_dir,self.subdir[i+1], ('cam_'+str(j+1))))
                image_list.sort()
                for k in range(len(image_list)):
                    self.images_path.append(image_list[k])
        self.test_set = torch.zeros(130, 3, 112, 112)  # len=130
        for i in range(len(self.labels)):
            label_test = cv2.imread(os.path.join(self.label_path, self.labels[i]))
            label_test = cv2.resize(label_test, (112, 112))
            # print('the size of label_test:{}'.format(label_test.shape))
            label_test = transforms.ToTensor()(label_test)
            self.test_set[i] = label_test

    def __len__(self):
        return 15*len(self.labels)  # number of Videos for train or test

    def __getitem__(self, idx):  # get a stochastic clip (length=5) of video
        '''
        :param idx:
        :return:
            images:     Tensor    seqtrain * 3 * width * height
            label_map:  Tensor    45 * 45 * (class+1) * seqtrain
            center_map: Tensor    1 * 368 * 368
        '''
        image_path = self.images_path[idx]  # '001_cam1_2.jpg'
        label, cam, distance = image_path.split('_')  # '001', 'cam1', '2'
        cam_list = list(cam)
        cam_list.insert(3, '_')
        cam_idx = ''.join(cam_list)  # 'cam_1'
        label_idx = int(label)
        distance = distance.split('.')[0]
        distance_idx = int(distance)
        img = cv2.imread(os.path.join(self.data_dir, self.subdir[distance_idx], cam_idx, image_path))  # [120,120,3]
        # print("this is path:{}".format(os.path.join(self.data_dir, self.subdir[distance_idx],cam_idx)))
        img = cv2.resize(img, (112, 112))  # [112,112,3]
        img = transforms.ToTensor()(img)
        label = cv2.imread(os.path.join(self.label_path,self.labels[label_idx-1]))  # [120,120,3]
        label = cv2.resize(label, (112,112))  # [112,112,3]
        label = transforms.ToTensor()(label)
        return img, label, self.test_set

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


# test case
#data = Penn_Data(data_dir='Penn_Action/', transform=transform1)
#images, label_map, center_map = data[1]
