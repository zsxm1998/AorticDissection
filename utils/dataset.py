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
    def __init__(self, img_dir, depth, transform):
        self.img_dir = img_dir
        self.depth = depth
        self.transform = transform
        self.labels = [label for label in listdir(img_dir) if isdir(join(img_dir, label))].sort()
        self.datas = []
        for i, label in enumerate(self.labels):
            img_list =  list(filter(lambda x: not x.startswith('.') and isfile(join(img_dir, label, x)), listdir(join(img_dir, label)))).sort()
            il_len = len(img_list)
            for j in range(il_len):
                if j + depth > il_len:
                    break
                nl = re.split('[_.]', img_list[j])
                nld = re.split('[_.]', img_list[j+depth-1])
                assert len(nl) == 4 and len(nld) == 4, 'Format of image file name is wrong.'
                if nl[0] != nld[0] or nl[1] != nld[1] or int(nl[2])+depth-1 != int(nld[2]):
                    continue
                self.datas.append([[join(img_dir, label, img_list[k]) for k in range(j, j+depth)], i])
        train_log.info(f'Creating dataset with {len(self.datas)} examples')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        label = torch.tensor(self.datas[i][1], dtype=torch.LongTensor)
        img_path_list = self.datas[i][0]
        img_list = []
        for img_path in img_path_list:
            img = Image.open(img_path)
            img_list.append(self.transform(img))
        imgs = torch.stack(img_list, dim=1)
        return imgs, label
            


        




