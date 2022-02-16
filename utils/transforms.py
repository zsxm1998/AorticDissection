import random
import numbers
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple, List, Optional
from torch import Tensor
from PIL import Image, ImageFilter


__all__ = ['Resize3D', 'CenterCrop3D', 'RandomHorizontalFlip3D', 'RandomVerticalFlip3D', 'ColorJitter3D', 'RandomRotation3D', 'ToTensor3D']


def _interpolation_modes_from_int(i: int) -> T.InterpolationMode:
    inverse_modes_mapping = {
        0: T.InterpolationMode.NEAREST,
        2: T.InterpolationMode.BILINEAR,
        3: T.InterpolationMode.BICUBIC,
        4: T.InterpolationMode.BOX,
        5: T.InterpolationMode.HAMMING,
        1: T.InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


class Resize3D(torch.nn.Module):
    def __init__(self, size, interpolation=T.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, tuple)):
            raise TypeError("Size should be int or tuple. Got {}".format(type(size)))
        if isinstance(size, tuple) and len(size) not in (1, 2):
            raise ValueError("If size is a tuple, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, imgs):
        return [TF.resize(img, self.size, self.interpolation, self.max_size, self.antialias) for img in imgs]

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class CenterCrop3D(torch.nn.Module):
    @staticmethod
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, tuple) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    def __init__(self, size):
        super().__init__()
        self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, imgs):
        return [TF.center_crop(img, self.size) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip3D(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if torch.rand(1) < self.p:
            return [TF.hflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip3D(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if torch.rand(1) < self.p:
            return [TF.vflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ColorJitter3D(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, apply_idx=None):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.apply_idx = apply_idx

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, imgs):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                imgs = [TF.adjust_brightness(img, brightness_factor) if self.apply_idx is None or i in self.apply_idx else img for i, img in enumerate(imgs)]
            elif fn_id == 1 and contrast_factor is not None:
                imgs = [TF.adjust_contrast(img, contrast_factor) if self.apply_idx is None or i in self.apply_idx else img for i, img in enumerate(imgs)]
            elif fn_id == 2 and saturation_factor is not None:
                imgs = [TF.adjust_saturation(img, saturation_factor) if self.apply_idx is None or i in self.apply_idx else img for i, img in enumerate(imgs)]
            elif fn_id == 3 and hue_factor is not None:
                imgs = [TF.adjust_hue(img, hue_factor) if self.apply_idx is None or i in self.apply_idx else img for i, img in enumerate(imgs)]

        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation3D(torch.nn.Module):
    @staticmethod
    def _check_sequence_input(x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
        if not isinstance(x, (list, tuple)):
            raise TypeError("{} should be a sequence of length {}.".format(name, msg))
        if len(x) not in req_sizes:
            raise ValueError("{} should be sequence of length {}.".format(name, msg))

    @staticmethod
    def _setup_angle(x, name, req_sizes=(2, )):
        if isinstance(x, numbers.Number):
            if x < 0:
                raise ValueError("If {} is a single number, it must be positive.".format(name))
            x = [-x, x]
        else:
            RandomRotation3D._check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]

    def __init__(
        self, degrees, interpolation=T.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = self._setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            self._check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (list, tuple, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, imgs):
        fill = self.fill
        if isinstance(imgs[0], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * TF.get_image_num_channels(imgs[0])
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return [TF.rotate(img, angle, self.resample, self.expand, self.center, fill) for img in imgs]

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string


class ToTensor3D:
    def __call__(self, imgs):
        return [TF.to_tensor(img) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GaussianResidual(torch.nn.Module):
    def __init__(self, ksize, sigma, before=True, flag_3d=False, apply_idx=None):
        super().__init__()
        if isinstance(ksize, int):
            assert ksize <= 0 or ksize % 2 == 1, f'ksize should be either odd or <= 0, but ksize={ksize}.'
            self.ksize = (ksize, ksize)
        elif isinstance(ksize, (tuple, list)):
            assert len(ksize) == 2, f'length of ksize should be 2 but {len(ksize)}.'
            self.ksize = tuple(ksize)
        else:
            raise TypeError("ksize should be a single number or a list/tuple with length 2.")
        self.sigma = sigma
        self.before = before
        self.flag_3d = flag_3d
        self.apply_idx = apply_idx

    def forward(self, img):
        if self.before:
            if self.flag_3d:
                raise NotImplementedError('在前和3d组合没有实现。')
            else:
                img = np.asarray(img)
                img_gau = cv2.GaussianBlur(img, self.ksize, self.sigma)
                res = img - img_gau
                res = np.stack((img, res), axis=-1)
                return Image.fromarray(res)
        else:
            if self.flag_3d:
                for i, im in enumerate(img):
                    if self.apply_idx is None or i in self.apply_idx:
                        im = im.permute((1,2,0)).squeeze(-1).numpy()
                        g1_x = cv2.Sobel(im, -1, dx=1, dy=0, ksize=self.ksize[0])
                        g1_y = cv2.Sobel(im, -1, dx=0, dy=1, ksize=self.ksize[0])
                        g1 = np.abs(g1_x) + np.abs(g1_y)
                        #g1 = cv2.blur(g1, self.ksize)
                        res = np.stack((im, g1), axis=-1)
                        img[i] = torch.from_numpy(res).permute((2,0,1))
                return img
            else:
                # img_gau = img.permute((1,2,0)).numpy()
                # img_gau = cv2.GaussianBlur(img_gau, self.ksize, self.sigma)
                # img_gau = torch.from_numpy(img_gau).unsqueeze(-1).permute((2,0,1))
                # res = img - img_gau
                # return torch.cat([img, res], dim=0)
                img = img.permute((1,2,0)).squeeze(-1).numpy()
                #img_gau = cv2.GaussianBlur(img, self.ksize, self.sigma)
                #g2 = cv2.Laplacian(img_gau, -1, ksize=self.ksize[0])
                g1_x = cv2.Sobel(img, -1, dx=1, dy=0, ksize=self.ksize[0])
                g1_y = cv2.Sobel(img, -1, dx=0, dy=1, ksize=self.ksize[0])
                g1 = np.abs(g1_x) + np.abs(g1_y)
                #g1 = cv2.blur(g1, self.ksize)
                #res = img_gau - g2
                res = np.stack((img, g1), axis=-1)
                return torch.from_numpy(res).permute((2,0,1))