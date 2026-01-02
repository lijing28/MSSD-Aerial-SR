import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation
from ..myutils.img_util import convert_image_to_fn
from ..myutils.misc import exists
from utils import util

import imageio
import pandas as pd

def load_img(filepath):
    img = imageio.imread(filepath)
    # img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor

class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="datasets/pngtxt", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
                is_train=True
        ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.is_train = is_train

        # self.degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        print(f"===> center_crop={center_crop} image_size={image_size}")
        self.img_preproc = transforms.Compose([
            #transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_paths = []
        folders = os.listdir(pngtxt_dir)
        self.img_paths = sorted(glob.glob(f'{pngtxt_dir}/*.jpg'))
        # for folder in folders:
        #     self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/*.png'))[:])

        df = pd.read_csv('/data0/llj/dataset/AID/real_world/train_HR.csv')
        self.caption = dict(zip(df['name'], df['caption']))

        upscale_factor = 4
        blur_kernel = 21
        blur_type = "aniso_gaussian"
        sig_min = 0.2
        sig_max = 4.0
        lambda_min = 0.2
        lambda_max = 4.0
        noise = 25
        self.degrade = util.SRMDPreprocessing(
            upscale_factor,
            kernel_size=blur_kernel,
            blur_type=blur_type,
            sig_min=sig_min,
            sig_max=sig_max,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            noise=noise,
        )


    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # # load image
        img_path = self.img_paths[index]
        txt_path = img_path.replace(".png", ".txt")
    
        image = Image.open(img_path).convert('RGB')

        image = self.crop_preproc(image)
    
        example["pixel_values"] = self.img_preproc(image)

        if self.control_type is not None:
            if self.control_type == 'realisr':
                # B, N, C, H, W
                # GT_image_t, LR_image_t = self.degradation.degrade_process(np.asarray(image)/255., resize_bak=self.resize_bak)
                GT_image_t = np2Tensor(np.asarray(image)).unsqueeze(0).unsqueeze(0) # image[1,1,3,h,w]
                if self.is_train:
                    LR_image_t, bkernel = self.degrade(GT_image_t, random=True)  # degradation LR_image_t [1, 1, 3, 200, 200]
                else:
                    LR_image_t, bkernel = self.degrade(GT_image_t, random=False)  # degradation LR_image_t [1, 1, 3, 200, 200]
                LR_image_t = LR_image_t.squeeze() / 255.0
                GT_image_t = GT_image_t.squeeze() / 255.0

                example["conditioning_pixel_values"] = LR_image_t # [b,c,h,w] range [0,1]
                example["pixel_values"] = GT_image_t * 2.0 - 1.0 # [b,c,h,w] range [-1,1]
            elif self.control_type == 'grayscale':
                image = np.asarray(image.convert('L').convert('RGB'))/255.
                example["conditioning_pixel_values"] = torch.from_numpy(image).permute(2,0,1)
            else:
                raise NotImplementedError

        # fp = open(txt_path, "r")
        # caption = fp.readlines()[0]
        # caption = ""
        img_n = img_path.split("/")[-1]
        caption = self.caption[img_n]
        # print(img_n, caption)
        if self.tokenizer is not None:
            example["input_ids"] = self.tokenize_caption(caption).squeeze(0)
        # fp.close()
        
        # return example
        return  example["pixel_values"], "", example["input_ids"], example["conditioning_pixel_values"]

    def __len__(self):
        return len(self.img_paths)