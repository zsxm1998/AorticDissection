# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import os
import sys
from pathlib import Path
import re
from os import listdir
from os.path import join, isdir, isfile
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

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
                assert len(nl) == 4, f'Format of image file name "{img_list[j]}" is wrong.'
                # nld = re.split('[_.]', img_list[j + (depth-1)*step])
                # assert len(nl) == 4 and len(nld) == 4, 'Format of image file name is wrong.'
                # if nl[0] != nld[0] or nl[1] != nld[1] or int(nl[2])+(depth-1)*step != int(nld[2]):
                #     continue
                s = 0
                group_list = []
                for k in range(j, j + (depth-1)*step+1):
                    nlk = re.split('[_.]', img_list[k])
                    assert len(nlk) == 4, f'Format of image file name "{img_list[k]}" is wrong.'
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
        train_log.info(f'Creating dataset with {len(self.datas)} examples. Depth:{depth}, Step:{step}, Residual:{residual}')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][1], dtype=torch.long)
        img_path_list = self.datas[i][0]
        img_list = []
        # for img_path in img_path_list:
        #     img = Image.open(img_path)
        #     img = self.transform(img)
        #     if self.residual and img_list:
        #         res = img - img_list[-1]
        #         res = (res + 1) / 2
        #         img_list.append(res)
        #     img_list.append(img)
        for img_path in img_path_list:
            img_list.append(Image.open(img_path))
        img_list = self.transform(img_list)
        if self.residual:
            for i in range(1, len(img_list)):
                res = img_list[i] - img_list[i-1]
                res = (res + 1) / 2
                img_list.append(res)
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


class LabelSampler(Sampler[int]):
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.labels = [data_source[i][1] for i in range(len(data_source))]
        self.unique_labels = np.unique(self.labels)
        self.labels_index_dict = {self.unique_labels[i]: [] for i in range(len(self.unique_labels))}
        for i, label in enumerate(self.labels):
            self.labels_index_dict[label].append(i)
        self.shortest_label = -1
        self.shortest_label_len = sys.maxsize
        for k, v in self.labels_index_dict.items():
            if len(v) < self.shortest_label_len:
                self.shortest_label = k
                self.shortest_label_len = len(v)
        assert self.shortest_label != -1

    def __iter__(self):
        sample_list = []
        for k, v in self.labels_index_dict.items():
            if k == self.shortest_label:
                sample_list.extend(v)
            else:
                sample_list += np.random.choice(v, self.shortest_label_len, False).tolist()
        if self.shuffle:
            np.random.shuffle(sample_list)
        return iter(sample_list)

    def __len__(self):
        return self.shortest_label_len * len(self.unique_labels)


class AortaDataset3DCenter(Dataset):
    def __init__(self, img_dir, transform, depth, step=1, residual=False, supcon=False):
        self.img_dir = img_dir
        self.transform = transform
        assert depth % 2 == 1, 'depth should be odd number.'
        self.depth = depth
        self.step = step
        self.residual = residual
        self.supcon = supcon
        self.labels = sorted([label for label in listdir(img_dir) if isdir(join(img_dir, label))])
        self.datas = []
        for i, label in enumerate(self.labels):
            img_list = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, label, x)), listdir(join(img_dir, label)))))
            il_len = len(img_list)
            for j in range(il_len):
                nl = re.split('[_.]', img_list[j])
                assert len(nl) == 4 or len(nl) == 5, f'Format of image file name "{img_list[j]}" is wrong.'
                if len(nl) == 5:
                    continue
                group_list = [img_list[j]]
                half = s = depth // 2
                for k in range(j-1, j-step*(half)-1, -1):
                    if k < 0 or k >= il_len:
                        for _ in range(s):
                            group_list.insert(0, group_list[0])
                        break
                    nlk = re.split('[_.]', img_list[k])
                    if nl[0] != nlk[0] or nl[1] != nlk[1]:
                        for _ in range(s):
                            group_list.insert(0, group_list[0])
                        break
                    offset = int(nl[2]) - int(nlk[2])
                    if offset % step == 0:
                        for _ in range(offset//step-(half-s)):
                            group_list.insert(0, img_list[k])
                            s -= 1
                            if s == 0:
                                break
                    if s == 0:
                        break
                s = depth // 2
                for k in range(j+1, j+step*(half)+1):
                    if k < 0 or k >= il_len:
                        for _ in range(s):
                            group_list.insert(0, group_list[0])
                        break
                    nlk = re.split('[_.]', img_list[k])
                    if nl[0] != nlk[0] or nl[1] != nlk[1]:
                        for _ in range(s):
                            group_list.append(group_list[-1])
                        break
                    offset = int(nlk[2]) - int(nl[2])
                    if offset % step == 0:
                        for _ in range(offset//step-(half-s)):
                            group_list.append(img_list[k])
                            s -= 1
                            if s == 0:
                                break
                    if s == 0:
                        break
                assert len(group_list) == depth, f'depth wrong: {img_list[j]}'
                self.datas.append([[join(img_dir, label, img) for img in group_list], i])
        train_log = logging.getLogger('train_log')
        train_log.info(f'Creating dataset with {len(self.datas)} examples. Depth:{depth}, Step:{step}, Residual:{residual}')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][1], dtype=torch.long)
        img_path_list = self.datas[i][0]
        img_list = []
        for img_path in img_path_list:
            img_list.append(Image.open(img_path))
        if self.supcon:
            img_list1 = self.transform(img_list)
            if self.residual:
                for i in range(1, len(img_list1)):
                    res = img_list1[i] - img_list1[i-1]
                    res = (res + 1) / 2
                    img_list1.append(res)
            imgs1 = torch.stack(img_list1, dim=1)
            img_list2 = self.transform(img_list)
            if self.residual:
                for i in range(1, len(img_list2)):
                    res = img_list2[i] - img_list2[i-1]
                    res = (res + 1) / 2
                    img_list2.append(res)
            imgs2 = torch.stack(img_list2, dim=1)
            return [imgs1, imgs2], label
        else:
            img_list = self.transform(img_list)
            if self.residual:
                for i in range(1, len(img_list)):
                    res = img_list[i] - img_list[i-1]
                    res = (res + 1) / 2
                    img_list.append(res)
            imgs = torch.stack(img_list, dim=1)
            return imgs, label


class MultiChannel(Dataset):
    def __init__(self, img_dir, c2_dir, c3_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = sorted([label for label in listdir(img_dir) if isdir(join(img_dir, label))])
        self.datas = []
        not_exist_list = []
        for i, label in enumerate(self.labels):
            img_list = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, label, x)), listdir(join(img_dir, label)))))
            for img in img_list:
                c1 = join(img_dir, label, img)
                c2 = join(c2_dir, label, img)
                c3 = join(c3_dir, label, img)
                if not os.path.exists(c2) or not os.path.exists(c3):
                    if not os.path.exists(c2):
                        not_exist_list.append(c2)
                    if not os.path.exists(c3):
                        not_exist_list.append(c3)
                    continue
                self.datas.append([[c1, c2, c3], i])
        train_log = logging.getLogger('train_log')
        train_log.info(f'Creating dataset with {len(self.datas)} examples.')
        train_log.info(f'{len(not_exist_list)} imgs do not exist: {not_exist_list}')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][1], dtype=torch.long)
        img_path_list = self.datas[i][0]
        img_list = []
        for img_path in img_path_list:
            img_list.append(Image.open(img_path))
        img = Image.merge('RGB', img_list)
        img = self.transform(img)
        return img, label


class MaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.labels = sorted([label for label in listdir(img_dir) if isdir(join(img_dir, label))])
        self.datas = []
        for i, label in enumerate(self.labels):
            img_list = sorted(list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, label, x)), listdir(join(img_dir, label)))))
            for img in img_list:
                img_path = join(img_dir, label, img)
                mask_path = join(mask_dir, img)
                if not os.path.exists(mask_path):
                    mask_path = None
                self.datas.append([img_path, mask_path, i])
        train_log = logging.getLogger('train_log')
        train_log.info(f'Creating dataset with {len(self.datas)} examples.')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][2], dtype=torch.long)
        img = Image.open(self.datas[i][0])
        if self.datas[i][1] is None:
            mask = Image.fromarray(np.ones((img.height, img.width), dtype=np.uint8)*255)
        else:
            mask = Image.open(self.datas[i][1])
        img, mask = self.transform([img, mask])
        return img, mask, label