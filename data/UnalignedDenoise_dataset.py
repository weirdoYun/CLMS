# from os import makedev
import os.path
import torch
import SimpleITK as sitk
# from typing import AsyncIterable
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transform
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
import random
import cv2
import torchvision.transforms as transforms
import numpy as np
from skimage import transform



class UnalignedDenoiseDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.targetLibPath = opt.targetLibPath
        self.targetLib = torch.load(self.targetLibPath)
        self.normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        self.A_size = len(self.targetLib['slide'])
        self.augLib = torch.load(opt.augLibPath)


    def niiImageRead(self, path):
        itkimg = sitk.ReadImage(path)
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg_zeros = np.zeros((npimg.shape[0], npimg.shape[1], 3)).astype(np.float32)
        for i in range(0, 3):
            npimg_zeros[:, :, i] = npimg
        return npimg_zeros

    def imageTrans_ForCycleGAN(self, image):
        resizedImage = cv2.resize(image,(384,384))
        x, y = resizedImage.shape[0], resizedImage.shape[1]

        randx = np.random.randint(0, x - 128)
        randy = np.random.randint(0, y - 128)
        croppedImage = resizedImage[randx:randx + 128, randy:randy + 128,:]

        if random.random() < 0.5:
            croppedImage = cv2.flip(croppedImage, 0)

        if random.random() < 0.5:
            croppedImage = cv2.flip(croppedImage, 1)
        ImageNormalize = transforms.Compose([transforms.ToTensor(), self.normalize])

        return ImageNormalize(croppedImage)

    def trans_ForSeg(self, image):
        ImageNormalize = transforms.Compose([transforms.ToTensor(), self.normalize])
        if random.random() < 0.5:
            x, y = image.shape[0], image.shape[1]

            randx = np.random.randint(0, x - 192)
            randy = np.random.randint(0, y - 192)
            image = image[randx:randx + 192, randy:randy + 192]
        image = cv2.resize(image, (384, 384))

        if random.random() < 0.5:
            image = cv2.flip(image, 0)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        image = ImageNormalize(image)
        return image


    def __getitem__(self, index):
        t_path = self.targetLib['slide'][index % self.A_size]
        t_img = self.niiImageRead(t_path)
        tp_img = self.imageTrans_ForCycleGAN(t_img)

        t_path_ori = self.targetLib['slide'][index % self.A_size]
        t_img = self.niiImageRead(t_path_ori)
        t_img = self.trans_ForSeg(t_img)

        aug_path = self.augLib['slide'][index % len(self.augLib['slide'])]
        aug_image = self.niiImageRead(aug_path)
        aug_image = self.trans_ForSeg(aug_image)
        return {'tp_img':tp_img,  't_path':t_path, 't_img': t_img, 'aug_image': aug_image}

    def __len__(self):
        return len(self.augLib['slide'])


