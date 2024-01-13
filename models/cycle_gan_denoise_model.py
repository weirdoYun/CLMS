import copy

import torch
import itertools
import torch.optim as optim

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .model_seg import *
import torch.nn as nn
import random
flag_amp = False

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        # self.bce = nn.BCELoss()
        self.smooth = smooth
        self.reduction = reduction
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        shape = input.shape
        N = shape[0]
        C = shape[1]
        if C != 1:
            input = F.softmax(input.clone(), dim=1).max(dim=1, keepdim=True)[0]

        input_flat = input.clone().contiguous().view(N, -1).float()
        target_flat = target.clone().contiguous().view(N, -1).float()

        # input_flat = 1 - input_flat
        # target_flat = 1 - target_flat

        intersection = torch.sum(torch.mul(input_flat, target_flat), dim=1)

        scores = (2. * intersection + self.smooth) / (
                    torch.sum(input_flat, dim=1) + torch.sum(target_flat, dim=1) + self.smooth)
        loss_d = 1.0 - scores

        if self.reduction == 'mean':
            loss_d = loss_d.mean()
        elif self.reduction == 'sum':
            loss_d = loss_d.sum()
        else:
            loss_d = loss_d

        # loss = self.bce(input_flat, target_flat.detach())

        return loss_d


import torch
import torch.nn as nn





def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module, mom):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mean = torch.zeros(module.running_mean.data.shape)
        self.var = torch.zeros(module.running_var.data.shape)
        self.momentum = mom

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        # print('input[0].size(): ', input[0].size())
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        self.mean = self.mean.detach() * (1 - self.momentum) + mean.cpu() * self.momentum
        self.var = self.var.detach() * (1 - self.momentum) + var.cpu() * self.momentum


        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(torch.sqrt(module.running_var.data.cpu() + 1e-6) - torch.sqrt(self.var.cpu() + 1e-6), 2) + torch.norm(module.running_mean.data.cpu() - self.mean.cpu(), 2)
        # print('r_feature: ', type(r_feature))
        r_feature = torch.sum(r_feature)
        self.r_feature = r_feature


    def close(self):
        self.hook.remove()




class TVLoss_new(nn.Module):
    def __init__(self,TVLoss_weight_spatial=1, TVLoss_weight_channel=1):
        super(TVLoss_new,self).__init__()
        self.TVLoss_weight_spatial = TVLoss_weight_spatial
        self.TVLoss_weight_channel = TVLoss_weight_channel


    def forward(self, x_realA, x_fakeB):
        diff = x_fakeB - x_realA

        h_x = x_realA.size()[2]
        w_x = x_realA.size()[3]

        '''spatial'''

        h_tv_realA = x_realA[:,:,1:,:]-x_realA[:,:,:h_x-1,:]
        w_tv_realA = x_realA[:,:,:,1:]-x_realA[:,:,:,:w_x-1]

        h_tv_fakeB = x_fakeB[:,:,1:,:]-x_fakeB[:,:,:h_x-1,:]
        w_tv_fakeB = x_fakeB[:,:,:,1:]-x_fakeB[:,:,:,:w_x-1]

        h_tv_diff = diff[:, :, 1:, :] - diff[:, :, :h_x - 1, :]
        w_tv_diff = diff[:, :, :, 1:] - diff[:, :, :, :w_x - 1]

        h_mask = torch.where((h_tv_realA * h_tv_fakeB)<= 0, torch.ones_like(h_tv_realA), torch.zeros_like(h_tv_realA))
        w_mask = torch.where((w_tv_realA * w_tv_fakeB)<= 0, torch.ones_like(w_tv_realA), torch.zeros_like(w_tv_realA))

        loss_spatial =  torch.sum(torch.abs(h_tv_diff * h_mask) ) / torch.sum(h_mask + 1e-6) + torch.sum(torch.abs(w_tv_diff * w_mask ) ) / (torch.sum(w_mask) + 1e-6)

        '''channel'''
        rg_tv_realA, rb_tv_realA, gb_tv_realA = x_realA[:,0,:,:] - x_realA[:,1,:,:], x_realA[:,0,:,:] - x_realA[:,2,:,:], x_realA[:,1,:,:] - x_realA[:,2,:,:]
        rg_tv_fakeB, rb_tv_fakeB, gb_tv_fakeB = x_fakeB[:,0,:,:] - x_fakeB[:,1,:,:], x_fakeB[:,0,:,:] - x_fakeB[:,2,:,:], x_fakeB[:,1,:,:] - x_fakeB[:,2,:,:]

        rg_tv_diff, rb_tv_diff, gb_tv_diff = diff[:,0,:,:] - diff[:,1,:,:], diff[:,0,:,:] - diff[:,2,:,:], diff[:,1,:,:] - diff[:,2,:,:]

        rg_mask = torch.where((rg_tv_realA * rg_tv_fakeB) <= 0, torch.ones_like(rg_tv_realA), torch.zeros_like(rg_tv_realA))
        rb_mask = torch.where((rb_tv_realA * rb_tv_fakeB) <= 0, torch.ones_like(rb_tv_realA), torch.zeros_like(rb_tv_realA))
        gb_mask = torch.where((gb_tv_realA * gb_tv_realA) <= 0, torch.ones_like(gb_tv_realA), torch.zeros_like(gb_tv_realA))
        loss_channel = torch.sum(torch.abs(rg_tv_diff * rg_mask) ) / torch.sum(rg_mask + 1e-6) + torch.sum(torch.abs(rb_tv_diff * rb_mask)) / (torch.sum(rb_mask) + 1e-6) + torch.sum(torch.abs(gb_tv_diff * gb_mask)) / (torch.sum(gb_mask) + 1e-6)
        return loss_spatial * self.TVLoss_weight_spatial + loss_channel * self.TVLoss_weight_channel




class CycleGANDenoiseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [ 'cycle_AB', 'idt_AB', 'seg_A_BCE', 'seg_A_Dice', 'seg_aug_BCE', 'seg_aug_Dice', 'seg_A_style', 'constraintGA_r_feature', 'constraintGA_newTV']
        visual_names_A = [ 'real_A', 'fake_B']


        self.visual_names = visual_names_A   # combine visualizations for A and B
        self.visual_names.append('A_image_ori')
        self.visual_names.append('A_image_fakeB')
        self.visual_names.append('rec_image_A')

        self.visual_names.append('pred_mask_A_rec')
        self.visual_names.append('PseudoLabel_A')


        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', '_seg']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        self.net_seg = Deeplab(num_classes = 1)
        chcekpointsPath = opt.sourceModel_checkpointsPath

        checkpoints = torch.load(chcekpointsPath)
        self.net_seg.load_state_dict(checkpoints)
        self.net_seg.cuda()
        if len(self.gpu_ids)> 1:
            print(self.gpu_ids)
            self.net_seg = torch.nn.DataParallel(self.net_seg, self.gpu_ids)


        self.net_predLabel = Deeplab(num_classes = 1)
        chcekpointsPath = opt.sourceModel_checkpointsPath

        checkpoints = torch.load(chcekpointsPath)
        self.net_predLabel.load_state_dict(checkpoints)
        self.net_predLabel.apply(fix_bn)

        self.net_predLabel.cuda()
        if len(self.gpu_ids) > 1:
            print(self.gpu_ids)
            self.net_predLabel = torch.nn.DataParallel(self.net_predLabel, self.gpu_ids)
        self.net_segForConstraintGA = copy.deepcopy(self.net_predLabel)


        self.net_seg.train()
        self.net_predLabel.eval()
        self.net_segForConstraintGA.eval()
        self.loss_BceLoss = nn.BCELoss()
        self.loss_diceLoss = DiceLoss()
        self.loss_TVLoss = TVLoss_new(opt.tvLoss_spatial_value, opt.tvLoss_channel_value)

        self.aug_value = opt.aug_value
        self.mom_value = opt.mom_value
        self.rebuild_full_value = opt.rebuild_full_value

        self.loss_r_feature_layers = []
        index = 0

        for module in self.net_segForConstraintGA.modules():
            if isinstance(module, nn.BatchNorm2d):
                if index < 30:
                    self.loss_r_feature_layers.append(DeepInversionFeatureHook(module, self.mom_value))
                index += 1


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-5)
            self.optimizer_seg = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.net_seg.parameters()), lr=opt.lr , betas=(opt.beta1, 0.999), weight_decay=5e-5)
            self.optimizer_constraintGA = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-5)

            # print('the type of self.optimizer_seg: ', self.optimizer_seg)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_seg)
            self.optimizers.append(self.optimizer_constraintGA)


            self.lr = opt.lr
            self.beat1 = opt.beta1

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['tp_img'].to(self.device)
        self.A_image_ori = input['t_img'].to(self.device)
        self.aug_image_ori = input['aug_image'].to(self.device)


    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

    # def forward_predLabel(self):
    #     with torch.no_grad():
    #         _, _, _, self.PseudoLabel_A = self.net_predLabel(self.A_image_ori)


    def forward_constraintGA(self):
        self.PseudoLabel_A = self.net_predLabel(self.A_image_ori)
        self.PseudoLabel_aug = self.net_predLabel(self.aug_image_ori)

        self.A_image_fakeB_forConstraintGA = self.netG_A(self.A_image_ori)
        self.rec_A_forConstraintGA = self.netG_B(self.A_image_fakeB_forConstraintGA)
        self.output_constraintGA = self.net_segForConstraintGA(self.A_image_fakeB_forConstraintGA)



    def forward_ASeg(self):
        self.A_image_fakeB = self.netG_A(self.A_image_ori)
        self.rec_image_A = self.netG_B(self.A_image_fakeB)
        self.pred_mask_A_rec = self.net_seg(self.rec_image_A)


        self.pred_mask_aug_rec = self.net_seg(self.aug_image_ori)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        # Identity loss
        if lambda_idt > 0:
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            # self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_G =  self.loss_cycle_A  + self.loss_idt_B
        self.loss_G.backward()


    def backward_constraintGA(self):
        self.loss_constraintGA_r_feature = sum([mod.r_feature for (idx, mod) in enumerate(self.loss_r_feature_layers[0:10])])
        self.loss_seg_A_style = self.criterionCycle(self.rec_A_forConstraintGA, self.A_image_ori) * self.rebuild_full_value
        self.loss_constraintGA_newTV = self.loss_TVLoss(x_realA =  self.A_image_ori, x_fakeB = self.A_image_fakeB_forConstraintGA)
        self.loss_constraintGA = self.loss_constraintGA_r_feature + self.loss_seg_A_style + self.loss_constraintGA_newTV

        self.loss_constraintGA.backward(retain_graph=True)




    def backward_seg_A(self):
        self.loss_seg_A_BCE = self.loss_BceLoss( self.pred_mask_A_rec,  self.PseudoLabel_A.detach())
        self.loss_seg_A_Dice = self.loss_diceLoss(self.pred_mask_A_rec, self.PseudoLabel_A.detach())

        self.loss_seg_aug_BCE = self.loss_BceLoss(self.pred_mask_aug_rec, self.PseudoLabel_aug.detach())
        self.loss_seg_aug_Dice = self.loss_diceLoss(self.pred_mask_aug_rec, self.PseudoLabel_aug.detach())

        self.loss_seg_A = self.loss_seg_A_BCE + self.loss_seg_A_Dice
        self.loss_seg_aug =  (self.loss_seg_aug_BCE + self.loss_seg_aug_Dice) * self.aug_value

        self.loss_seg = self.loss_seg_A + self.loss_seg_aug
        self.loss_seg.backward()




    def optimize_parameters(self, need_seg=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # self.forward_predLabel()
        self.set_requires_grad([self.netG_B, self.netG_A, self.net_seg], True)
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        ##=====================================================================
        self.loss_cycle_AB = self.loss_cycle_A
        if self.opt.lambda_identity > 0:
            self.loss_idt_AB =  self.loss_idt_B
       ##=====================================================================

        self.forward_constraintGA()
        self.optimizer_constraintGA.zero_grad()
        self.backward_constraintGA()
        self.optimizer_constraintGA.step()


        ##=====================================================================
        self.forward_ASeg()
        self.optimizer_seg.zero_grad()
        self.backward_seg_A()
        self.optimizer_seg.step()

        ##=====================================================================
        import torchvision.transforms as transform
        mask_transform = transform.Normalize((0.5,), (0.5,))
        with torch.no_grad():
            self.pred_mask_A_rec = mask_transform(self.pred_mask_A_rec[:, 0, :, :].unsqueeze(1).detach())
            self.PseudoLabel_A = mask_transform(self.PseudoLabel_A[:, 0, :, :].unsqueeze(1).detach())

        ##=====================================================================



# self.loss_names = ['D_AB', 'G_AB', 'cycle_AB', 'idt_AB', 'seg_AB', 'seg_fake_AB']