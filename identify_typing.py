import os
import re
import glob
import shutil
import argparse
import sys
import traceback
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import pydicom
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms as T
from tqdm import tqdm

from yolov5_detect import run_yolo_detect
from models.resnet3d import generate_model
from utils.datasets import AortaTest, AortaTest3D

PAD_NUM = 4
POSITIVE_THRESHOLD = 4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in filter(lambda x: x.endswith('.dcm'), os.listdir(path))]
    slices.sort(key=lambda x: float(x.InstanceNumber))
    return slices


def generate_image(patient_folder, ww, wl):
    lower_b, upper_b = int(wl - ww // 2), int(wl + ww // 2)
    image_path = os.path.join(patient_folder, f'images_{ww}_{wl}')
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)

    ct = load_scan(os.path.join(patient_folder))

    name = os.path.basename(patient_folder)
    for i in range(len(ct)):
        img = ct[i].pixel_array.astype(np.int16)
        intercept = ct[i].RescaleIntercept
        slope = ct[i].RescaleSlope
        if slope != 1:
            img = (slope * img.astype(np.float64)).astype(np.int16)
        img += np.int16(intercept)
        img = np.clip(img, lower_b, upper_b)
        img = ((img-lower_b)/(upper_b-lower_b)*255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(image_path, f'{name}_{i:04d}.png'))
    return image_path


def find_branch(label_path):
    label_file_list = sorted(glob.glob(os.path.join(label_path, '*.txt')))
    label_dict = {}
    pre = -1
    candidate_branch = []
    for i, label_txt in enumerate(label_file_list):
        with open(label_txt, 'r') as txtf:
            lines = txtf.readlines()
        assert 0 < len(lines) < 3, f'{label_txt}'
        index = int(re.split('[_.]', os.path.basename(label_txt))[-2])
        elem = []
        if i < 0.5*len(label_file_list) and pre > 0 and len(lines) > pre:
            candidate_branch.append(index)
        pre = len(lines)
        if len(lines) == 1:
            aorta = lines[0].split()
            assert len(aorta) == 5, f'{label_txt}'
            elem.append([float(aorta[1]), float(aorta[2]), float(aorta[3]), float(aorta[4])])
        else:
            line0, line1 = lines[0].split(), lines[1].split()
            assert len(line0) == 5 and len(line1) == 5, f'{label_txt}'
            line0, line1 = list(map(lambda x: float(x), line0)), list(map(lambda x: float(x), line1))
            dx, dy = abs(line0[1] - line1[1]), abs(line0[2] - line1[2])
            dim = 1 if dx > dy else 2
            if line0[dim] > line1[dim]:
                aorta, branch = line0, line1
            else:
                aorta, branch = line1, line0
            elem.append([aorta[1], aorta[2], aorta[3], aorta[4]])
            elem.append([branch[1], branch[2], branch[3], branch[4]])
        label_dict[index] = elem
    assert len(candidate_branch) > 0, f'{label_path}'
    keys_list = list(label_dict.keys())
    min_idx, max_idx = keys_list[0], keys_list[-1]
    branch_start, branch_end, b_len = -1, -1, -1
    for cbs in candidate_branch:
        head = cbs - 1
        while head >= min_idx:
            res = label_dict.get(head)
            if res is not None and len(res) > 1:
                break
            head -= 1
        head += 1
        tail = cbs
        pad = PAD_NUM
        while tail <= max_idx:
            res = label_dict.get(tail)
            if res is not None:
                if len(res) < 2:
                    pad -= 1
                    if pad == 0:
                        break
                else:
                    pad = PAD_NUM
            tail += 1
        tmp_b_len = tail - head
        if tmp_b_len > b_len:
            b_len = tmp_b_len
            branch_start = cbs
            branch_end = tail-PAD_NUM+1 if tail <= max_idx else tail
    return branch_start, branch_end, label_dict, min_idx, max_idx


def calc_coordinate(height, width, label):
    x, y, w, h = label[0], label[1], label[2], label[3]
    w, h = int(width*w), int(height*h)
    w, h = max(w, h), max(w, h)
    return int(width*x-w/2), int(height*y-h/2), int(width*x+w/2+1), int(height*y+h/2+1)


def crop_image(image_path, branch_start, branch_end, label_dict, max_idx):
    base_path = os.path.dirname(image_path)
    base_name = os.path.basename(base_path)
    crop_path = os.path.join(base_path, 'crops')
    j_path = os.path.join(crop_path, 'j')
    s_path = os.path.join(crop_path, 's')
    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)
    os.makedirs(j_path)
    os.mkdir(s_path)
    jx, jy, sx, sy = -1, -1, -1, -1
    for i in range(branch_start, branch_end):
        label = label_dict.get(i)
        if label is None:
            continue
        img = Image.open(os.path.join(image_path, f'{base_name}_{i:04d}.png'))
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]
        if i == branch_start:
            jx, jy, sx, sy = label[0][0], label[0][1], label[1][0], label[1][1]
            x1, y1, x2, y2 = calc_coordinate(height, width, label[0])
            crop = img[y1:y2, x1:x2]
            crop = Image.fromarray(crop)
            crop.save(os.path.join(j_path, f'{base_name}_{i:04d}.png'))
            x1, y1, x2, y2 = calc_coordinate(height, width, label[1])
            crop = img[y1:y2, x1:x2]
            crop = Image.fromarray(crop)
            crop.save(os.path.join(s_path, f'{base_name}_{i:04d}.png'))
        else:
            js_flag = []
            crop_list = []
            min_dis_list = []
            for j in range(len(label)):
                dis_j, dis_s = sqrt((jx-label[j][0])**2+(jy-label[j][1])**2), sqrt((sx-label[j][0])**2+(sy-label[j][1])**2)
                if dis_s < dis_j:
                    js_flag.append('s')
                    min_dis_list.append(dis_s)
                else:
                    js_flag.append('j')
                    min_dis_list.append(dis_j)
                x1, y1, x2, y2 = calc_coordinate(height, width, label[j])
                crop = img[y1:y2, x1:x2]
                crop = Image.fromarray(crop)
                crop_list.append(crop)
            if len(crop_list) == 1:
                if js_flag[0] == 'j':
                    crop_list[0].save(os.path.join(j_path, f'{base_name}_{i:04d}.png'))
                    jx, jy = label[0][0], label[0][1]
                else:
                    crop_list[0].save(os.path.join(s_path, f'{base_name}_{i:04d}.png'))
                    sx, sy = label[0][0], label[0][1]
            else:
                if js_flag[0] == js_flag[1]:
                    idx = 0 if min_dis_list[0] < min_dis_list[1] else 1
                    if js_flag[idx] == 'j':
                        crop_list[idx].save(os.path.join(j_path, f'{base_name}_{i:04d}.png'))
                        jx, jy = label[idx][0], label[idx][1]
                    else:
                        crop_list[idx].save(os.path.join(s_path, f'{base_name}_{i:04d}.png'))
                        sx, sy = label[idx][0], label[idx][1]
                else:
                    for idx in range(len(crop_list)):
                        if js_flag[idx] == 'j':
                            crop_list[idx].save(os.path.join(j_path, f'{base_name}_{i:04d}.png'))
                            jx, jy = label[idx][0], label[idx][1]
                        else:
                            crop_list[idx].save(os.path.join(s_path, f'{base_name}_{i:04d}.png'))
                            sx, sy = label[idx][0], label[idx][1]
    for i in range(branch_end, max_idx):
        label = label_dict.get(i)
        if label is None:
            continue
        img = Image.open(os.path.join(image_path, f'{base_name}_{i:04d}.png'))
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]
        crop_list = []
        dis_list = []
        for j in range(len(label)):
            dis_j = sqrt((jx-label[j][0])**2+(jy-label[j][1])**2)
            dis_list.append(dis_j)
            x1, y1, x2, y2 = calc_coordinate(height, width, label[j])
            crop = img[y1:y2, x1:x2]
            crop = Image.fromarray(crop)
            crop_list.append(crop)
        idx = 0 if len(crop_list) == 1 or dis_list[0] < dis_list[1] else 1
        crop_list[idx].save(os.path.join(j_path, f'{base_name}_{i:04d}.png'))
        jx, jy = label[idx][0], label[idx][1]

    return j_path, s_path


def create_net(device,
               n_channels=1,
               n_classes=1,
               load_model=False,
               flag_3d=True
               ):
    if flag_3d:
        net = generate_model(34, n_channels=n_channels, n_classes=n_classes, conv1_t_size=3)
    else:
        net = models.resnet34(pretrained=False)
        net.n_channels, net.n_classes = n_channels, n_classes
        net.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        #net.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    if load_model:
        net.load_state_dict(torch.load(load_model, map_location=device))
    net.to(device=device)
    net.eval()
    return net


@torch.no_grad()
def aorta_classify(model, device, aorta_path, transform, flag_3d=True):
    if flag_3d:
        dateset = AortaTest3D(aorta_path, transform, depth=11, step=3)
    else:
        dateset = AortaTest(aorta_path, transform)
    dataloader = DataLoader(dateset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    n_data = len(dateset)
    pred_ori_list = []
    pred_list = []

    with tqdm(total=n_data, desc=aorta_path, unit='img') as pbar:
        for imgs in dataloader:
            assert imgs.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            imgs = imgs.to(device=device, dtype=torch.float32)

            categories_pred = model(imgs)

            if model.n_classes > 1:
                pred = torch.softmax(categories_pred, dim=1)
                pred_ori_list += pred.tolist()
                pred = pred.argmax(dim=1)
                pred_list.extend(pred.tolist())
            else:
                pred = torch.sigmoid(categories_pred)
                pred_ori_list += pred.squeeze(1).tolist()
                pred = (pred > 0.5).float()
                pred_list.extend(pred.squeeze(-1).tolist())

            pbar.update(imgs.shape[0])

    print(f"{aorta_path}:", pred_list)
    positive_count = 0
    for res in pred_list:
        if res == 1:
            positive_count += 1
            if positive_count == POSITIVE_THRESHOLD:
                return True
        else:
            positive_count = 0
    return False


def delete_temp_dir(image_path):
    parents_path = os.path.dirname(image_path)
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    if os.path.exists(os.path.join(parents_path, 'labels')):
        shutil.rmtree(os.path.join(parents_path, 'labels'))
    # if os.path.exists(os.path.join(parents_path, 'pred_images')):
    #     shutil.rmtree(os.path.join(parents_path, 'pred_images'))
    if os.path.exists(os.path.join(parents_path, 'crops')):
        shutil.rmtree(os.path.join(parents_path, 'crops'))


def main(source,
         yolo_weight,
         resnet_weight,
         window_width=600,
         window_level=200,
         flag_3d = True
         ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_net(device, load_model=resnet_weight, flag_3d=flag_3d)
    transform = T.Compose([
        T.Resize(51), # 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
        T.CenterCrop(51), # 从图片中间切出img_size*img_size的图片
        T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
        #T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
    ])
    source_list = []
    for fp in os.listdir(source):
        if os.path.isdir(os.path.join(source, fp)):
            source_list.append(os.path.join(source, fp, '1'))
        elif fp.endswith('.dcm'):
            source_list = [source]
            break
    
    result_list = []
    for patient in source_list:
        try:
            image_path = generate_image(patient, window_width, window_level)
            label_path = run_yolo_detect(yolo_weight, image_path, imgsz=256, max_det=2, save_img=True)
            branch_start, branch_end, label_dict, min_idx, max_idx = find_branch(label_path)
            print(f'branch_start: {branch_start}, branch_end: {branch_end}')
            j_path, s_path = crop_image(image_path, branch_start, branch_end, label_dict, max_idx)
            j_res = aorta_classify(model, device, j_path, transform, flag_3d=flag_3d)
            s_res = aorta_classify(model, device, s_path, transform, flag_3d=flag_3d)
            if s_res == True:
                print(f'{patient}分型: A')
                result_list.append('A')
            elif j_res == True:
                print(f'{patient}分型: B')
                result_list.append('B')
            else:
                print(f'{patient}分型: 阴性')
                result_list.append('N')
        except KeyboardInterrupt:
            image_path = os.path.join(patient, f'images_{window_width}_{window_level}')
            print(f'KeyboardInterrupt, deleteing temp dirs in: {os.path.dirname(image_path)}')
            delete_temp_dir(image_path)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except:
            traceback.print_exc()
        finally:
            delete_temp_dir(image_path)
    print(result_list)
    print(list(zip(source_list, result_list)))
        

def get_args():
    parser = argparse.ArgumentParser(description='Identify typing of aorta dissection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--source', type=str, required=True, help='Patient dicom files')
    parser.add_argument('-ww', '--window_width', type=int, default=600, help='window width')
    parser.add_argument('-wl', '--window_level', type=int, default=200, help='window level')
    parser.add_argument('-yw', '--yolo_weight', type=str, required=True, help='Yolov5 weight for aorta detection')
    parser.add_argument('-rw', '--resnet_weight', type=str, required=True, help='Resnet34 weight for classification')
    parser.add_argument('-t', '--flag_3d', action='store_true', help='Use 3D model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))