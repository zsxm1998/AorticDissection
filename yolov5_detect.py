# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, colorstr, is_ascii, non_max_suppression, \
    scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run_yolo_detect(weights,  # model.pt path(s)
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=256,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=2,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_img=False, # save detect result
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        ):

    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    else:
        imgsz *= 2 if len(imgsz) == 1 else 1

    # Directories
    save_dir = Path(source).parent
    pred_dir = save_dir / 'pred_images'
    label_dir = save_dir / 'labels'
    if save_img:
        pred_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        pred = model(img, augment=augment, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        t2 = time_sync()

        # Process predictions
        for det in pred:  # detections per image
            p, s, im0 = path, '', im0s.copy()

            p = Path(p)  # to Path
            pred_path = str(pred_dir / p.name)  # img.jpg
            txt_path = str(label_dir / p.stem)  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save results (image with detections)
            if save_img:
                im0 = annotator.result()
                cv2.imwrite(pred_path, im0)

    s = f"\n{len(list(label_dir.glob('*.txt')))} labels saved to {label_dir}" if save_txt else ''
    #print(f"Results saved to {colorstr('bold', pred_dir)}{s}")

    #print(f'Done. ({time.time() - t0:.3f}s)')
    return str(label_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=2, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-img', action='store_true', help='save detection results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    #run_yolo_detect(**vars(opt))
    source_dir = opt.source
    source_name = source_dir.split('/')[-1] if source_dir.split('/')[-1] != '' else source_dir.split('/')[-2]
    opt.project = os.path.join(opt.project, source_name)

    for patient in os.listdir(source_dir):
        if not os.path.isdir(os.path.join(source_dir, patient)):
            continue

        opt.source = os.path.join(source_dir, patient)
        opt.name = patient
        run_yolo_detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
