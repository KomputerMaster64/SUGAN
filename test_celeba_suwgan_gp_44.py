#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   ADA Implementation of the Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP)
#       Alternative implementation
#       2022.11.21
#       TEST
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Import relevant libraries for the WGAN-GP
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils

import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import numpy as np
import math
import sys

from torch import Tensor
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Relevant Directories
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CODE_DIR =      '/scratch/temp_44'
CODE_DIR_WGAN = '/home/manisha.padala/gan/WGAN-GP-DRAGAN-Celeba-Pytorch'
DATA_DIR =      '/scratch/temp_44/data'

#-------------------------------------
# Directory to save the images/samples
#-------------------------------------
sample_dir =    '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.22'
#   sample_dir contains the samples saved using the following method
# z = Variable(Tensor(np.random.normal(0, 1, (1, z_dim))))
## z = utils.cuda(Variable(torch.randn(bs, z_dim)))
## Generate a batch of images
# fake_imgs = G(z)
#-------------------------------------
sample_dir_2 =  '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.25_200000'
#   sample_dir_2 contains the samples saved using the following method
# z_sample = Variable(torch.randn(100, z_dim))
# z_sample = utils.cuda(z_sample)
# f_imgs_sample = (G(z_sample).data + 1) / 2.0
#-------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Defining the Gradient Penalty 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = utils.cuda(torch.rand(shape))
    z = x + alpha * (y - x)

    # gradient penalty
    z = utils.cuda(Variable(z, requires_grad=True))
    o = f(z)
    g = grad(o, z, grad_outputs=utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Using the GPU(s)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" gpu """
# gpu_id = [2]
gpu_id = [0]
utils.cuda_devices(gpu_id)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   WGAN-GP Hyperparameters
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" param """
epoch       = 50     # 50                                               # check the value to be loaded using the `.pth` file
n_epochs    = 100    # 50
batch_size  = 64     # 64
n_critic    = 5      # 5
lr          = 0.0002 # 0.0002
z_dim       = 100    # 100

sample_interval = 100
checkpoint_interval = 10
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Data Augmentation and/or Preprocessing
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" data """
crop_size = 108
re_size = 64
offset_height   = (218 - crop_size) // 2
offset_width    = (178 - crop_size) // 2
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(crop),
    transforms.ToPILImage(),
    # transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
    # https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
    transforms.Resize(size = (re_size, re_size), interpolation = Image.BICUBIC), # Resampling.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Configuration for the DataLoader
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# imagenet_data = dsets.ImageFolder('./data/img_align_celeba', transform=transform)
# imagenet_data = dsets.ImageFolder( DATA_DIR + '/img_align_celeba', transform=transform)
imagenet_data = dsets.ImageFolder( DATA_DIR + '/img_align_celeba_test', transform=transform)
data_loader = torch.utils.data.DataLoader(
                                            imagenet_data,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8 # num_workers = 4
                                        )
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Defining the Discriminator and Generator
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" model """
D = models_64x64.DiscriminatorWGANGP(3)
G = models_64x64.Generator(z_dim)
utils.cuda([D, G])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Defining the Optimiser
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" optimiser """
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Loading the checkpoint or saved data for resuming training
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" load checkpoint """
# ckpt_dir = './checkpoints/celeba_wgan_gp'
ckpt_dir = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.22'
# ckpt_dir_2 = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.22_2'
# ckpt_dir = '/home/manisha.padala/gan/CelebA_WGAN-GP_71_2022.11.21'

# utils.mkdir(ckpt_dir)

try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']                             # start_epoch = opt.epoch for reference
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:                                                     # NO EXCEPT STATEMENT REQUIRED
    print(' [*] No checkpoint!')
    start_epoch = 0
    print(' [*] Loading the `.pth` state dict!')
    G.load_state_dict(torch.load(   '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.22_2/generator_%d.pth'      % (epoch)))
    D.load_state_dict(torch.load(   '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.22_2/discriminator_%d.pth'  % (epoch)))


# if opt.epoch != 0:
#     G.load_state_dict(torch.load(   CODE_DIR + "/saved_models/%s/generator_%d.pth"      % (opt.dataset_name, opt.epoch)))
#     D.load_state_dict(torch.load(   CODE_DIR + "/saved_models/%s/discriminator_%d.pth"  % (opt.dataset_name, opt.epoch)))
# else:
#     # Initialize weights of the discriminator and generator
#     print("Initializing Weights")
#     # G.apply(weights_init_normal)
#     # D.apply(weights_init_normal)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Testing the WGAN-GP model
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# z_sample = Variable(torch.randn(100, z_dim))
# z_sample = utils.cuda(z_sample)
# batches_done = 0


# SELECT THE STARTING EPOCH 
# start_epoch = ckpt['epoch']
# start_epoch = epoch

if start_epoch != 0:
    
    # Set the Generator in inference mode for sampling
    G.eval()

    # Sample noise vector for Generator input
    # FIXED vs NON-FIXED Noise vector

    # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    for k in range(0,200001):
        
        # VERSION 1 SAMPLES in sample_dir = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.22'
        # # USE THIS
        # # z = Variable(Tensor(np.random.normal(0, 1, (1, z_dim))))
        # # z=utils.cuda(z)
        # # OR THIS
        # z = utils.cuda(Variable(torch.randn(1, z_dim)))
        # # Generate a batch of images
        # # fake_imgs = G(z)
        # fake_imgs = (G(z).data + 1) / 2.0 # MAYBE THIS IMPROVES THE CONTRAST
        # fake_fname = 'fake_'+str(k)+'.png' # ‘fake-{0:0=4d}.png’.format(i)
    
        # # PRINT MESSAGE
        # print('New image sample is ' + fake_fname)
        # # sample_dir = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.22'
        # torchvision.utils.save_image(fake_imgs, os.path.join(sample_dir, fake_fname), nrow=1) # nrow=8
        # # torchvision.utils.save_image(f_imgs.data, "/scratch/temp_44/images/CelebA_SUWGAN-GP_44_2022.11.22_2/%d.png" % batches_done, nrow=10) # fake_imgs = gen_igmgs; fake_imgs.data[:25], nrow = 5
        


        # VERSION 2 SAMPLES in sample_dir_2 = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.22_2'
        z_sample = Variable(torch.randn(1, z_dim))                                                # z_sample = Variable(torch.randn(100, z_dim))
        z_sample = utils.cuda(z_sample)
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        # sample_dir_2 = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.25_200000'
        fake_fname_2 = 'fake_'+str(k)+'.png' # ‘fake-{0:0=4d}.png’.format(i)
        print('New image sample is ' + fake_fname_2)
        torchvision.utils.save_image(f_imgs_sample, os.path.join(sample_dir_2, fake_fname_2), nrow=10) # nrow = 10

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################
################################################################################ END OF FILE ####################################################################################
#################################################################################################################################################################################    