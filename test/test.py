import os
import functools
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from DataLoader import Dataset
import models.model_seg as UNet
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn import metrics
import models.networks as networks
import cv2
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin) + 1e-6)


def calculate_avgDice(mask, prediction):
    sum_dice = 0
    for i in range(0, mask.shape[0]):
        tempMask, tempPred = mask[i], prediction[i]
        # singleImgDiceList.append(single_dice_coef(tempMask, tempPred))
        sum_dice += single_dice_coef(tempMask, tempPred)
    return sum_dice / mask.shape[0]




def clip_data(dataLib):
    imageList = dataLib['slide']
    labelList = dataLib['label']
    list_img_sim = []
    list_lbl_sim = []
    for i in range(len(labelList)):
        temp_lbl_np = cv2.imread(labelList[i])
        if temp_lbl_np.max() > 100:
            list_img_sim.append(imageList[i])
            list_lbl_sim.append(labelList[i])
    data_lib_sim = {}
    data_lib_sim['slide'] = list_img_sim
    data_lib_sim['label'] = list_lbl_sim
    return data_lib_sim


def func(TestLibPath,checkpointsPath, epoch):
    Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    Batchsize = 32
    num_worker = 4
    thre = 0.5

    TestLib = torch.load(TestLibPath)
    TestLib = clip_data(TestLib)

    TestDataSet = Dataset(TestLib, Normalize, 'test')
    TestDataLoader = DataLoader(TestDataSet, batch_size=Batchsize, shuffle=False, num_workers=num_worker)

    G_A = networks.UnetGenerator(3, 3, 7, 64, norm_layer=get_norm_layer(norm_type='instance'), use_dropout=False)
    weightPath_GA = os.path.join(checkpointsPath, epoch + '_net_G_A.pth')
    weight = torch.load(weightPath_GA)
    import collections
    dicts = collections.OrderedDict()
    for k, value in weight.items():
        if "module" in k:  # 去除命名中的module
            k = k.split(".")[1:]
            k = ".".join(k)
        dicts[k] = value

    weight = dicts
    G_A.load_state_dict(weight)
    G_A.cuda()

    segmodel = UNet.Deeplab(1)
    weightPath_seg = os.path.join(checkpointsPath, epoch + '_net__seg.pth')

    weight = torch.load(weightPath_seg)
    import collections
    dicts = collections.OrderedDict()
    for k, value in weight.items():
        if "module" in k:  # 去除命名中的module
            k = k.split(".")[1:]
            k = ".".join(k)
        dicts[k] = value
    weight = dicts
    segmodel.load_state_dict(weight)
    segmodel.cuda()


    mask1Dice_singleEpoch = 0
    with torch.no_grad():
        G_A.eval()
        segmodel.eval()
        GT1List = []
        Pred1List = []

        for i, (img, lab) in enumerate(TestDataLoader):
            img, lab = img.cuda(), lab.cuda()
            output_GA = G_A(img)
            output = segmodel(output_GA)
            output, lab = output.cpu(), lab.cpu()
            Pred1List.append(output[:,0,:,:].unsqueeze(1))
            GT1List.append(lab[:,0,:,:].unsqueeze(1))
        Pred1 = torch.cat(Pred1List, 0)
        GT1 = torch.cat(GT1List, 0)
        mask1Dice_singleEpoch = calculate_avgDice( GT1.numpy() > (50 / 100),Pred1.numpy() > (40 / 100))


    return thre, mask1Dice_singleEpoch


def func_withOutGA(TestLibPath, checkpointsPath, epoch):
    print('func_noAdaptation start!!')
    Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    Batchsize = 32
    num_worker = 4
    thre = 0.5
    TestLib = torch.load(TestLibPath)
    TestLib = clip_data(TestLib)



    TestDataSet = Dataset(TestLib, Normalize, 'test')
    TestDataLoader = DataLoader(TestDataSet, batch_size=Batchsize, shuffle=False, num_workers=num_worker)

    segmodel = UNet.Deeplab(1)
    weightPath_seg = os.path.join(checkpointsPath, epoch + '_net__seg.pth')

    weight = torch.load(weightPath_seg)
    segmodel.load_state_dict(weight)
    segmodel.cuda()


    with torch.no_grad():
        segmodel.eval()
        Pred1List = []
        GT1List = []
        for i, (img, lab) in enumerate(TestDataLoader):
            img, lab = img.cuda(), lab.cuda()
            output = segmodel(img)
            output, lab = output.cpu(), lab.cpu()
            Pred1List.append(output[:, 0, :, :].unsqueeze(1))
            GT1List.append(lab[:, 0, :, :].unsqueeze(1))
        Pred1 = torch.cat(Pred1List, 0)
        GT1 = torch.cat(GT1List, 0)
        mask1Dice_singleEpoch = calculate_avgDice( GT1.numpy() > (50 / 100),Pred1.numpy() > (40 / 100))

    return thre, mask1Dice_singleEpoch



if __name__ == '__main__':
        TestLibPath_target = ''
        TestLibPath_source = ''
        checkpointsPath = ''
        epoch = ''

        thre_target, dice_target = func(TestLibPath_target, checkpointsPath, epoch)
        thre_source, dice_source = func_withOutGA(TestLibPath_source, checkpointsPath, epoch)
