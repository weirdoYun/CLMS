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

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class randomGaussBlur(object):
    def __init__(self, radius=None):
        if radius is not None:
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return F.rotate(img, self.angle)



class Dataset(Dataset):

    def __init__(self, trainLib, normalize=None, mode = ''):
        self.imageList = trainLib['slide']
        self.labelList = trainLib['label']
        self.normalize = normalize
        self.mode = mode


    def niiImageRead(self, path):
        itkimg = sitk.ReadImage(path)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
        # print(npimg.shape)
        # npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)
        # print(npimg.shape)
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg_zeros = np.zeros((npimg.shape[0], npimg.shape[1], 3)).astype(np.float32)
        for i in range(0, 3):
            npimg_zeros[:, :, i] = npimg
        return npimg_zeros

    def trans_ForSeg(self, image, label1):
        ImageNormalize = transforms.Compose([transforms.ToTensor(), self.normalize])
        maskTrans = transforms.ToTensor()

        # image = self.image_padding(image, 3)
        # label1 = self.image_padding(label1, 1).astype(np.uint8)
        # label1_edge = self.image_padding(label1_edge, 1).astype(np.uint8)
        # label2 = self.image_padding(label2, 1).astype(np.uint8)
        # label2_edge = self.image_padding(label2_edge, 1).astype(np.uint8)

        if random.random() < 0.5:
            x, y = image.shape[0], image.shape[1]

            randx = np.random.randint(0, x - 192)
            randy = np.random.randint(0, y - 192)
            image = image[randx:randx + 192, randy:randy + 192]
            label1 = label1[randx:randx + 192, randy:randy + 192]
            # label1_edge = label1_edge[randx:randx + 128, randy:randy + 128]
            # label2 = label2[randx:randx + 128, randy:randy + 128]
            # label2_edge = label2_edge[randx:randx + 128, randy:randy + 128]

        image = cv2.resize(image, (384, 384))
        label1 = cv2.resize(label1, (384, 384))
        # label1_edge = cv2.resize(label1_edge, (256, 256))
        # label2 = cv2.resize(label2, (256, 256))
        # label2_edge = cv2.resize(label2_edge, (256, 256))

        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label1 = cv2.flip(label1, 0)
            # label1_edge = cv2.flip(label1_edge, 0)
            # label2 = cv2.flip(label2, 0)
            # label2_edge = cv2.flip(label2_edge, 0)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label1 = cv2.flip(label1, 1)
            # label1_edge = cv2.flip(label1_edge, 1)
            # label2 = cv2.flip(label2, 1)
            # label2_edge = cv2.flip(label2_edge, 1)

        image = ImageNormalize(image)
        label1 = maskTrans(label1)
        # label1_edge = maskTrans(label1_edge)
        # label2 = maskTrans(label2)
        # label2_edge = maskTrans(label2_edge)

        # label = torch.cat((label1, label2), 0)
        # label_edge = torch.cat((label1_edge, label2_edge), 0)

        label1[label1 > 0.5] = 1
        label1[label1 < 1] = 0

        # label_edge[label_edge > 0.5] = 1
        # label_edge[label_edge < 1] = 0
        return image, label1


    def __getitem__(self, index):
        if (self.mode == 'train'):
            imagePath = self.imageList[index]
            # image_idName =  imagePath.split('/')[-1][:-7]
            image = self.niiImageRead(imagePath)

            label1_path = self.labelList[index]
            # label2_path = self.dataForderPath + '/mask2/' + image_idName + '.png'

            label1 = cv2.imread(label1_path)
            # label2 = cv2.imread(label2_path)
            _, label1 = cv2.threshold(cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            # _, label2 = cv2.threshold(cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            image, label = self.trans_ForSeg(image, label1)
            return image, label

        if (self.mode == 'test'):
            imagePath = self.imageList[index]
            # image_idName =  imagePath.split('/')[-1][:-7]
            image = self.niiImageRead(imagePath)

            label1_path = self.labelList[index]
            label1 = cv2.imread(label1_path)
            # label2 = cv2.imread(label2_path)
            _, label1 = cv2.threshold(cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            # _, label2 = cv2.threshold(cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

            ImageNormalize = transforms.Compose([transforms.ToTensor(), self.normalize])
            maskTrans = transforms.ToTensor()
            # image = self.image_padding(image, 3)
            # label1 = self.image_padding(label1, 1).astype(np.uint8)
            # label2 = self.image_padding(label2, 1).astype(np.uint8)

            image = cv2.resize(image, (384, 384))
            label1 = cv2.resize(label1, (384, 384))
            # label2 = cv2.resize(label2, (256, 256))

            image = ImageNormalize(image)
            label1 = maskTrans(label1)
            # label2 = maskTrans(label2)
            # label = torch.cat((label1, label2), 0)
            label1[label1 > 0.5] = 1
            label1[label1 < 1] = 0

            return image, label1


    def __len__(self):
        return len(self.imageList)
