#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   ADA Implementation of the Supervised Unsupersised Wasserstein Generative Adversarial Networks with Gradient Penalty (SUWGAN-GP)
#       SUWGAN-GP
#       2022.11.22
#       blah
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
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch import device
from utils import cuda
import time

import os
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Relevant Directories
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CODE_DIR =      '/scratch/temp_44'
CODE_DIR_WGAN = '/home/manisha.padala/gan/WGAN-GP-DRAGAN-Celeba-Pytorch'
DATA_DIR =      '/scratch/temp_44/data'

DATA_TEST_img_align_celeba                      = DATA_DIR + '/img_align_celeba_test'               # 203_599 files img_align_celeba_test
DATA_TEST_CelebA_HQ_face_gender_dataset_test    = DATA_DIR + '/CelebA_HQ_face_gender_dataset_test'  # 30_000 files CelebA_HQ_face_gender_dataset_test
DATA_TEST_celeba_hq_train                       = DATA_DIR + '/celeba_hq/train'                     # 28_000 files celeba_hq_train
DATA_TEST_celeba_hq_val                         = DATA_DIR + '/celeba_hq/val'                       # 2_000 files celeba_hq_val

sample_dir      =  '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24'
sample_dir_2    =  '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24_2'        # NOT IN USE

# ALSO MENTIONED IN THE CHECKPOINT SECTION
ckpt_dir    = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24'    
ckpt_dir_2  = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24_2'  # NOT IN USE

fid_input_1 = ''
fid_input_2 = '/scratch/temp_44/images/CelebA_WGAN-GP_44_2022.11.23_DATASET'
fid_score_file = '/home/manisha.padala/gan/pytorch-fid/src/pytorch_fid/fid_score_44.py'
fid_score_output_file = '/scratch/temp_44/data/output_suwgan-gp_44_2022.11.24_FID.txt'
#  SAMPLE QUERY FOR THE FID SCORE 
# !python /home/manisha.padala/gan/pytorch-fid/src/pytorch_fid/fid_score.py --batch-size 16 --num-workers 8 --device cuda:0 /scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.22_2 /scratch/temp_44/data/img_align_celeba_fid
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
epochs      = 50     # 50
n_epochs    = 50     # 50
batch_size  = 64     # 64
n_critic    = 5      # 5
lr          = 0.0002 # 0.0002
z_dim       = 100    # 100

sample_interval = 100
checkpoint_interval = 10

lambda_gp = 10
lambda_classifier = 0.0002
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
# root = DATA_TEST_img_align_celeba                                                         ## YES
# DATA_TEST_img_align_celeba                                                                ## YES
# DATA_TEST_CelebA_HQ_face_gender_dataset_test
# DATA_TEST_celeba_hq_train                                                                 ## YES
# DATA_TEST_celeba_hq_val
imagenet_data = dsets.ImageFolder( DATA_DIR + '/img_align_celeba_test', transform=transform)
data_loader = torch.utils.data.DataLoader(
                                            imagenet_data,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8 # num_workers = 4
                                        )
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True )
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Defining ResNet model for the Gender Classifier
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The focus is on a transfer learning method.

genClass_model = models.resnet18(pretrained=True)
num_features = genClass_model.fc.in_features
genClass_model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
genClass_model = genClass_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(genClass_model.parameters(), lr=0.001, momentum=0.9) 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Get the pre-trained Gender Classifier in the `.pth` file
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# !wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EfAE05ATen9PopOPWzxLvMsBjzYIFOYaGaY2UpUcLETM7w?download=1 -O face_gender_classification_transfer_learning_with_ResNet18.pth
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Using the Pre-trained Gender CLassifier model
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Use the local path for the `.pth` file containing the saved model
# save_path = DATA_DIR + '/face_gender_classification_transfer_learning_with_ResNet18.pth'
save_path = '/home/manisha.padala/gan/2022.10.21_DCGAN_Scratch/data/face_gender_classification_transfer_learning_with_ResNet18.pth'
torch.save(genClass_model.state_dict(), save_path)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Load the trained model file to the ResNet architecture
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
genClass_model = models.resnet18(pretrained=True)
num_features = genClass_model.fc.in_features
genClass_model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
genClass_model.load_state_dict(torch.load(save_path))
genClass_model.to(device)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Put the trained ResNet model (of the Gender Classifier) into the evaluation/inference mode for "testing"
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

genClass_model.eval()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#################################################################################################################################################################################
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

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   Loading the checkpoint or saved data for resuming training
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" load checkpoint """
# ckpt_dir = './checkpoints/celeba_wgan_gp'
ckpt_dir    = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24'    
ckpt_dir_2  = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24_2'  # NOT IN USE
# utils.mkdir(ckpt_dir)

try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


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
#   Training the WGAN-GP model
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" run """
writer = tensorboardX.SummaryWriter( '/scratch/temp_44/summaries/CelebA_SUWGAN-GP_44_2022.11.24' )

z_sample = Variable(torch.randn(100, z_dim))
z_sample = utils.cuda(z_sample)

batches_done = 0

# SELECT THE STARTING EPOCH 
# start_epoch = ckpt['epoch']
# start_epoch = epochs
fname = 0
for epoch in range(start_epoch, n_epochs):
    criterion = nn.CrossEntropyLoss() # for the gender classifier
    for i, (imgs, labels) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable(imgs)
        imgs_2 = Variable(imgs)
        
        bs = imgs.size(0)
        z = Variable(torch.randn(bs, z_dim))
        imgs, z = utils.cuda([imgs, z])

        f_imgs = G(z)

        # train D
        r_logit = D(imgs)
        f_logit = D(f_imgs.detach())

        wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
        gp = gradient_penalty(imgs.data, f_imgs.data, D)
        d_loss = -wd + gp * 10.0

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        if step % n_critic == 0:
            # train G
            z = utils.cuda(Variable(torch.randn(bs, z_dim)))
            f_imgs = G(z)                                                # fake_imgs = Generator(z)

            #-------------------------------------------------
            #   Incorporation of the trained Gender Classifier
            #-------------------------------------------------
            start_time = time.time()
            imgs = imgs.to(device)
            labels = labels.to(device)
            genClass_outputs = genClass_model(f_imgs) # f_imgs = fake_imgs = gen_imgs
            # _, genClass_preds = torch.max(genClass_outputs, 1)
            # genClass_loss = criterion(genClass_outputs, labels)
            # genClass_preds = nn.functional.softmax(genClass_outputs, 1)
            criterion = nn.Softmax(dim = 1)
            genClass_preds_alt = criterion(genClass_outputs)
            genClass_loss_alt = (genClass_preds_alt[:,0].mean() - genClass_preds_alt[:,1].mean())**2
            # check the running loss code
            #-------------------------------------------------
            #-------------------------------------------------

            f_logit = D(f_imgs)
            # Original Generator Loss
            g_loss1 = -f_logit.mean()

            # lambda_classifier = ( g_loss1.detach() / genClass_loss ) / 10
            lambda_classifier = ( g_loss1.detach() / genClass_loss_alt ) / 10

            # g_loss = -f_logit.mean() + (lambda_classifier * genClass_loss)
            g_loss = -f_logit.mean() + (lambda_classifier * genClass_loss_alt)

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalars('G',
                               {"g_loss": g_loss.data.cpu().numpy()},
                               global_step=step)

            # print(
            # "[Epoch %3d] [Batch %5d/%5d] [D loss: %f] [G loss: %f]"
            # % (epoch, i, len(data_loader), d_loss.item(), g_loss.item())
            # )
            
            # Save output to file
            # https://stackoverflow.com/questions/36571560/directing-print-output-to-a-txt-file
            f = open( DATA_DIR + "/output_suwgan-gp_44_2022.11.24.txt", "a") # '/scratch/temp_44/data/output_suwgan-gp_44_2022.11.24.txt'
            print(
            "[Epoch %3d] [Batch %5d/%5d] [D loss: %f] [G loss: %f]"
            % (epoch, i, len(data_loader), d_loss.item(), g_loss.item()), file = f
            )
            f.close()
            
            # THE GENERATOR IS STILL IN TRAINING MODE G.train()
            if batches_done % sample_interval == 0:
                # torchvision.utils.save_image(f_imgs.data, CODE_DIR + "/images/%d.png" % batches_done, nrow=10, normalize=True) # fake_imgs = gen_igmgs; fake_imgs.data[:25], nrow = 5
                f_imgs_new = (f_imgs.data + 1) / 2.0
                torchvision.utils.save_image(f_imgs_new.data, "/scratch/temp_44/images/CelebA_SUWGAN-GP_44_2022.11.24_2/%d.png" % batches_done, nrow=10) # fake_imgs = gen_igmgs; fake_imgs.data[:25], nrow = 5
                #   /scratch/temp_44/images/CelebA_SUWGAN-GP_44_2022.11.24_2
                #   https://pytorch.org/vision/0.8/utils.html
            
                
            # Testing images for every 5th epoch
            if (epoch + 1) %  5 == 0:
            # if True:
                G.eval()
                for k in range(0,100001):
                    # VERSION 2 SAMPLES in sample_dir_2 = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24_2'
                    z_sample = Variable(torch.randn(1, z_dim)) # z_sample = Variable(torch.randn(100, z_dim))
                    z_sample = utils.cuda(z_sample)
                    f_imgs_sample = (G(z_sample).data + 1) / 2.0
                    # sample_dir  = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24'
                    fake_fname = 'fake_'+str(k)+'.png' # ‘fake-{0:0=4d}.png’.format(i)
                    # print('New image sample is ' + fake_fname)
                    # torchvision.utils.save_image(f_imgs_sample, os.path.join(sample_dir, fake_fname), nrow=1) # nrow = 1
                    torchvision.utils.save_image(f_imgs_sample, os.path.join('/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24', fake_fname), nrow=1) # nrow = 1
                print('100,000 Images Sampled. Now calculating the FID Scores')
                # fid_input_2 = '/scratch/temp_44/images/CelebA_WGAN-GP_44_2022.11.23_DATASET'
                # ctr = 6250, batch-size = 4, images = 25,000
                os.system('python /home/manisha.padala/gan/pytorch-fid/src/pytorch_fid/fid_score_44.py --batch-size 4 --num-workers 8 --device cuda:0 /scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24 /scratch/temp_44/images/CelebA_WGAN-GP_44_2022.11.23_DATASET')
                # python /home/manisha.padala/gan/pytorch-fid/src/pytorch_fid/fid_score.py --batch-size 16 --num-workers 8 --device cuda:0 /scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.23 /scratch/temp_44/images/CelebA_WGAN-GP_44_2022.11.23_DATASET
                shutil.rmtree('/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24')
                new_dir_path = '/scratch/temp_44/sample_dir/sample_dir_CelebA_SUWGAN-GP_44_2022.11.24'
                os.mkdir(new_dir_path)
            
            batches_done += n_critic

        # if (i + 1) % 1 == 0:
        #     print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(data_loader)))
        
        #-----------------------------------------------
        #   Alternate samples saved during training
        #-----------------------------------------------
        #   OUTSIDE TRAINING LOOP NOISE VECTOR USED
        #-----------------------------------------------
        #   z_sample = Variable(torch.randn(100, z_dim))
        #   z_sample = utils.cuda(z_sample)
        #-----------------------------------------------
        # if (i + 1) % 100 == 0:
        #     # Set the Generator into inference mode for sampling
        #     G.eval()
        #     f_imgs_sample = (G(z_sample).data + 1) / 2.0
        #     # save_dir = CODE_DIR + '/images/CelebA_WGAN-GP_44_2022.11.23'
        #     save_dir = '/scratch/temp_44/images/CelebA_SUWGAN-GP_44_2022.11.23'
        #     # utils.mkdir(save_dir)
        #     torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)
    print(
        "[Epoch %3d] [Batch %5d/%5d] [D loss: %f] [G loss: %f]"
        % (epoch, i, len(data_loader), d_loss.item(), g_loss.item())
    )


    utils.save_checkpoint({ 
                            'epoch': epoch + 1,
                            'D': D.state_dict(),
                            'G': G.state_dict(),
                            'd_optimizer': d_optimizer.state_dict(),
                            'g_optimizer': g_optimizer.state_dict()},
                            '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),       #   ckpt_dir = '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24'                                                                                                
                            max_keep=2
    )

    # opt.checkpoint_interval and opt.dataset_name are values associated with the parser
    # if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
    #     # Save model checkpoints
    #                                                                             # '/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24_2'
    #     torch.save(G.state_dict(),  "/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24_2/generator_%d.pth"      % (epoch))
    #     torch.save(D.state_dict(),  "/scratch/temp_44/saved_models/CelebA_SUWGAN-GP_44_2022.11.24_2/discriminator_%d.pth"  % (epoch))
    #     # torch.save(G.state_dict(),  CODE_DIR + "/saved_models/%s/generator_%d.pth"      % (opt.dataset_name, epoch))
    #     # torch.save(D.state_dict(),  CODE_DIR + "/saved_models/%s/discriminator_%d.pth"  % (opt.dataset_name, epoch))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################
################################################################################ END OF FILE ####################################################################################
#################################################################################################################################################################################    