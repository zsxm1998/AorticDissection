# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import os
from pathlib import Path
import re
from os import listdir
from os.path import join, isdir, isfile
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.augmentations import letterbox

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class AortaDataset3D(Dataset):
    def __init__(self, img_dir, transform, depth, step=1, residual=False):
        self.img_dir = img_dir
        self.transform = transform
        self.depth = depth
        self.step = step
        self.residual = residual
        self.labels = sorted([label for label in listdir(img_dir) if isdir(join(img_dir, label))])
        self.datas = []
        for i, label in enumerate(self.labels):
            img_list = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, label, x)), listdir(join(img_dir, label)))))
            il_len = len(img_list)
            for j in range(il_len - (depth-1)*step):
                nl = re.split('[_.]', img_list[j])
                assert len(nl) == 4, 'Format of image file name is wrong.'
                # nld = re.split('[_.]', img_list[j + (depth-1)*step])
                # assert len(nl) == 4 and len(nld) == 4, 'Format of image file name is wrong.'
                # if nl[0] != nld[0] or nl[1] != nld[1] or int(nl[2])+(depth-1)*step != int(nld[2]):
                #     continue
                s = 0
                group_list = []
                for k in range(j, j + (depth-1)*step+1):
                    nlk = re.split('[_.]', img_list[k])
                    assert len(nlk) == 4, 'Format of image file name is wrong.'
                    if nl[0] != nlk[0] or nl[1] != nlk[1]:
                        break
                    if int(nl[2]) + s*step == int(nlk[2]):
                        group_list.append(img_list[k])
                        s += 1
                        if s == depth:
                            break
                if s == depth:
                    self.datas.append([[join(img_dir, label, img) for img in group_list], i])
                # self.datas.append([[join(img_dir, label, img_list[k]) for k in range(j, j + (depth-1)*step+1)], i])
        train_log = logging.getLogger('train_log')
        train_log.info(f'Creating dataset with {len(self.datas)} examples. Depth:{depth}, Step:{step}')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][1], dtype=torch.long)
        img_path_list = self.datas[i][0]
        img_list = []
        for img_path in img_path_list:
            img = Image.open(img_path)
            img = self.transform(img)
            if self.residual and img_list:
                res = img - img_list[-1]
                res = (res + 1) / 2
                img_list.append(res)
            img_list.append(img)
        imgs = torch.stack(img_list, dim=1)
        return imgs, label

class AortaTest(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.datas = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, x)), listdir(img_dir))))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        img = Image.open(join(self.img_dir, self.datas[i]))
        img = self.transform(img)
        return img

class AortaTest3D(Dataset):
    def __init__(self, img_dir, transform, depth, step=1, residual=False):
        self.img_dir = img_dir
        self.transform = transform
        self.depth = depth
        self.step = step
        self.residual = residual
        self.files = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, x)), listdir(img_dir))))
        self.datas = []
        il_len = len(self.files)
        for j in range(il_len - (depth-1)*step):
            nl = re.split('[_.]', self.files[j])
            assert len(nl) == 3, 'Format of image file name is wrong.'
            s = 0
            group_list = []
            for k in range(j, j + (depth-1)*step+1):
                nlk = re.split('[_.]', self.files[k])
                assert len(nlk) == 3, 'Format of image file name is wrong.'
                assert nl[0] == nlk[0], 'Name of crops are different!'
                if int(nl[1]) + s*step == int(nlk[1]):
                    group_list.append(self.files[k])
                    s += 1
                    if s == depth:
                        break
            if s == depth:
                self.datas.append([join(img_dir, img) for img in group_list])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        img_path_list = self.datas[i]
        img_list = []
        for img_path in img_path_list:
            img = Image.open(img_path)
            img = self.transform(img)
            if self.residual and img_list:
                res = img - img_list[-1]
                #res = (res + 1) / 2
                img_list.append(res)
            img_list.append(img)
        imgs = torch.stack(img_list, dim=1)
        return imgs