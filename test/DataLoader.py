from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import torch
import torchvision.transforms.functional  as F
import cv2
import numpy as np
from PIL import Image, ImageFilter
import os
import SimpleITK as sitk




class Dataset(Dataset):

    def __init__(self, trainLib, normalize=None, mode = ''):
        self.imageList = trainLib['slide']
        self.labelList = trainLib['label']
        self.normalize = normalize
        self.mode = mode


    def niiImageRead(self, path):
        itkimg = sitk.ReadImage(path)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
        npimg = npimg.astype(np.float32)
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg_zeros = np.zeros((npimg.shape[0], npimg.shape[1], 3)).astype(np.float32)
        for i in range(0, 3):
            npimg_zeros[:, :, i] = npimg
        return npimg_zeros


    def __getitem__(self, index):

        if (self.mode == 'test'):
            imagePath = self.imageList[index]
            image = self.niiImageRead(imagePath)

            label1_path = self.labelList[index]
            label1 = cv2.imread(label1_path)
            _, label1 = cv2.threshold(cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

            ImageNormalize = transforms.Compose([transforms.ToTensor(), self.normalize])
            maskTrans = transforms.ToTensor()
    

            image = cv2.resize(image, (384, 384))
            label1 = cv2.resize(label1, (384, 384))

            image = ImageNormalize(image)
            label1 = maskTrans(label1)
            label1[label1 > 0.5] = 1
            label1[label1 < 1] = 0

            return image, label1


    def __len__(self):
        return len(self.imageList)
