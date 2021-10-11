import os
import re
from os import listdir
from os.path import join, isdir, isfile

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .print_log import train_log

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
                img_list.append(img - img_list[-1])
            img_list.append(img)
        imgs = torch.stack(img_list, dim=1)
        return imgs, label
            


        




