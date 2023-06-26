# %%
from __future__ import print_function  # Ensures print function compatibility with Python 2.x
import argparse  # Library for parsing command-line arguments
import os  # Library for interacting with the operating system
import copy  # Library for creating object copies
import numpy as np  # Numerical computing library
import random  # Library for generating random numbers

import torch  # Core library for PyTorch
import torch.nn as nn  # Library for defining neural network components
import torch.nn.parallel  # Library for parallelizing operations on multiple GPUs
import torch.backends.cudnn as cudnn  # Interface to the cuDNN library for GPU optimizations
import torch.optim as optim  # Library for optimization algorithms
import torch.utils.data  # Tools for working with datasets in PyTorch
import torchvision.datasets as dset  # Datasets provided by torchvision
import torchvision.transforms as transforms  # Transformations for image preprocessing
import torchvision.utils as vutils  # Utility functions for visualizing images
from torch.autograd import Variable  # Provides automatic differentiation for tensors
import torch.nn.functional as F  # Library for various activation functions and loss functions

from functions import *
from model import *

# %%
# Importing the necessary libraries
import argparse

# Creating an argument parser
parser = argparse.ArgumentParser()

# Adding arguments with their default values and descriptions
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--dataset', default='fashion', help='Dataset to use: cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='Path to the dataset')
parser.add_argument('--num_workers', type=int, help='Number of data loading workers', default=2)
parser.add_argument('--is_continue', type=int, default=1, help='Use pre-trained model')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
parser.add_argument('--image_size', type=int, default=32, help='Height/width of the input image to the network')
parser.add_argument('--latent_size', type=int, default=256, help='Size of the latent vector')
parser.add_argument('--num_epochs', type=int, default=55, help='Number of epochs to train for')
parser.add_argument('--weight_cycle_consistency', type=float, default=1.0, help='Weight of Cycle Consistency')
parser.add_argument('--W', type=float, default=1.0, help='Wake')
parser.add_argument('--N', type=float, default=1.0, help='NREM')
parser.add_argument('--R', type=float, default=1.0, help='REM')
parser.add_argument('--epsilon', type=float, default=0.0, help='Amount of noise in the wake latent space')
parser.add_argument('--num_filters', type=int, default=64, help='Filters factor')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='Probability of dropout')
parser.add_argument('--learning_rate_generator', type=float, default=0.0002, help='Learning rate for the generator')
parser.add_argument('--learning_rate_discriminator', type=float, default=0.0002, help='Learning rate for the discriminator')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
parser.add_argument('--lmbd', type=float, default=0.5, help='convex combination factor for REM')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--output_folder', default='output', help='Folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')



# Parsing the command-line arguments
opt, unknown = parser.parse_known_args()

# Set the number of iterations to the number of epochs
opt.niter = opt.num_epochs

# Assign the value of latent_size based on opt.latent_size
latent_size = opt.latent_size

# Printing the parsed arguments
print(opt)


# %% [markdown]
#  1. The code begins by importing the necessary library, **`argparse`**, which provides a way to parse command-line arguments.
# 
# 2. An argument parser object is created using **`argparse.ArgumentParser()`**. This object will handle the parsing of command-line arguments.
# 
# 3. Various arguments are added to the parser using the **`add_argument()`** method. Each argument has a unique name, a default value, and a description.
# 
# 4. The command-line arguments are parsed using **`parser.parse_known_args()`**, which returns two values: **`opt`** (containing the parsed argument values) and **`unknown`** (containing any unrecognized arguments).
# 
# 5. The parsed argument values are stored in the **`opt`** object.
# 
# 6. Finally, the values of the parsed arguments are printed using **`print(opt)`**.
# 
# This code allows you to run the program with different options and values from the command line. Each option represents a specific setting or parameter that can be customized. The **`argparse`** library handles the parsing of these options, and the **`opt`** object stores the values for further use within the program. Printing the **`opt`** object provides a summary of the parsed argument values, allowing you to verify the settings before running the main logic of the program.

# %%
# Specify the GPU ID if using only 1 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

# where to save samples and training curves
dir_files = './results/'+opt.dataset+'/'+opt.outf
# where to save model
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf


# Create the directories if they don't exist
try:
    os.makedirs(dir_files)
except OSError:
    pass
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass


# Set the device to CUDA if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset and get relevant information
dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize)

# Create a data loader for loading the dataset in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                         num_workers=int(opt.num_workers), drop_last=True)


# %% [markdown]
# 1. The code sets the environment variable **`CUDA_VISIBLE_DEVICES`** to the GPU ID specified in **`opt.gpu_id`**. This ensures that only the specified GPU is used when running the program.
# 
# 2. The code defines two directory paths: **`dir_files`** for saving samples and training curves, and **`dir_checkpoint`** for saving models.
# 
# 3. The code attempts to create the directories specified by **`dir_files`** and **`dir_checkpoint`**. If the directories already exist, the **`OSError`** exception is caught and passed.
# 
# 4. The code checks if CUDA is available and assigns the device accordingly. If CUDA is available, the device is set to the GPU with ID 0; otherwise, it is set to CPU.
# 
# 5. The code calls the **`get_dataset()`** function to load the dataset specified in **`opt.dataset`** from the directory **`opt.dataroot`**, and also obtains the **`unorm`** (normalization) and **`img_channels`** (number of image channels) values.
# 
# 6. Finally, the code creates a **`DataLoader`** object called **`dataloader`**, which allows iterating over the dataset in batches. The batch size is specified by **`opt.batchSize`**, and the data is shuffled and loaded in parallel using **`opt.num_workers`** worker processes. The **`drop_last`** option ensures that the last incomplete batch is dropped if its size is less than **`opt.batchSize`**.
# 
# This code prepares the necessary setup before training the neural network, such as configuring the device, setting up directories for saving results, and loading the dataset.

# %% [markdown]
# 

# %%
# Define and assign values to hyperparameters
num_gpus = int(opt.num_gpus)
latent_dim = int(opt.latent_size)
batch_size = opt.batch_size

# Instantiate generator and discriminator networks
generator = Generator(num_gpus, latent_dim=latent_dim, ngf=opt.num_filters, img_channels=img_channels)
generator.apply(initialize_weights)
discriminator = Discriminator(num_gpus, latent_dim=latent_dim, ndf=opt.num_filters, img_channels=img_channels, dropout_prob=opt.dropout_prob)
discriminator.apply(initialize_weights)


# Move networks to the GPU
generator.to(device)
discriminator.to(device)

# %% [markdown]
# 1. The code sets up some hyperparameters, such as the number of GPUs available (num_gpus), the size of the latent space (latent_dim), and the batch size (batch_size).
# 
# 2. Two neural network models are instantiated: the generator (named "generator") and the discriminator (named "discriminator").
# 
# 3. The generator is an instance of the Generator class, which takes the number of GPUs, latent dimension, number of filters, and number of channels as arguments.
# 
# 4. The discriminator is an instance of the Discriminator class, which takes similar arguments as the generator along with a dropout probability.
# 
# 5. Weight initialization is applied to both the generator and discriminator using the weights_init function.
# 
# 6. The generator and discriminator models are moved to the specified device (e.g., GPU) using the to() method.
# 
# 7. This ensures that the computations for these models will be performed on the GPU if available, which can significantly speed up training.

# %%
# Set up optimizers for discriminator and generator
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=opt.learning_rate_discriminator, betas=(opt.beta1, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=opt.learning_rate_generator, betas=(opt.beta1, 0.999))

# Initialize lists to store losses
d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []

# %% [markdown]
# The code sets up optimizers for the discriminator and generator models.
# 
# 1. The discriminator_optimizer uses the Adam optimizer and takes the discriminator parameters, learning rate (lrD), and beta values as arguments.
# 
# 2. The generator_optimizer uses the Adam optimizer and takes the generator parameters, learning rate (lrG), and beta values as arguments.
# 
# 3. Lists are initialized to store various types of losses during training.
# 
# 4. **discriminator_losses:** Stores the losses of the discriminator model.
# 
# 5. **generator_losses:** Stores the losses of the generator model.
# 
# 6. **real_losses:** Stores losses related to real images.
# 
# 7. **fake_losses:** Stores losses related to fake/generated images.
# 
# 8. **kl_losses:** Stores Kullback-Leibler divergence losses, which are often used in variational autoencoders (VAEs) or other generative models.

# %%
if os.path.exists(dir_checkpoint+'/trained.pth') and opt.is_continue:
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator_optimizer.load_state_dict(checkpoint['g_optim'])
    discriminator_optimizer.load_state_dict(checkpoint['d_optim'])
    d_losses = checkpoint.get('d_losses', [float('inf')])
    g_losses = checkpoint.get('g_losses', [float('inf')])
    r_losses_real = checkpoint.get('r_losses_real', [float('inf')])
    r_losses_fake = checkpoint.get('r_losses_fake', [float('inf')])
    kl_losses = checkpoint.get('kl_losses', [float('inf')])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# %%
# Define the loss functions
discriminator_criterion = nn.BCELoss()  # Binary Cross Entropy Loss for the discriminator
reconstruction_criterion = nn.MSELoss()  # Mean Squared Error Loss for reconstruction

# Create tensor placeholders
discriminator_label = torch.zeros(opt.batch_size, dtype=torch.float32, device=device)
real_label_value = 1.0
fake_label_value = 0

evaluation_noise = torch.randn(batch_size, latent_dim, device=device)


# %% [markdown]
# # 
# 
# 1. The code defines two loss functions: **`discriminator_criterion`** and **`reconstruction_criterion`**. The **`BCELoss`** (Binary Cross Entropy Loss) is used for the discriminator, and the **`MSELoss`** (Mean Squared Error Loss) is used for reconstruction.
# 
# 2. The variables **`dis_criterion`** and **`rec_criterion`** are changed to **`discriminator_criterion`** and **`reconstruction_criterion`**, respectively. 
# 
# 3. The variable names **`dis_label`**, **`real_label_value`**, **`fake_label_value`**, and **`eval_noise`** are changed to **`discriminator_label`**, **`real_label_value`**, **`fake_label_value`**, and **`evaluation_noise`**, respectively.
# 
# 4. The order and structure of the code remain unchanged as they are necessary for defining the loss functions and creating tensor placeholders.
# 
# Overall, this code snippet defines the loss functions for the discriminator and reconstruction tasks. The **`BCELoss`** is commonly used for binary classification tasks, such as determining whether an image is real or fake. The **`MSELoss`** is used for measuring the pixel-wise difference between the input and reconstructed images. The tensor placeholders are created to hold the discriminator labels, real and fake label values, and noise for evaluation.

# %%
# Enable anomaly detection during training (optional)
# torch.autograd.set_detect_anomaly(True)

# Training loop
for epoch in range(len(discriminator_losses), opt.niter):
    
    # Initialize lists to store losses and other metrics
    store_loss_D = []
    store_loss_G = []
    store_loss_R_real = []
    store_loss_R_fake = []
    store_norm = []
    store_kl = []

    # Iterate over the data batches
    for i, data in enumerate(dataloader, 0):

        ############################
        # Wake (W)
        ###########################

        # Discrimination wake
        discriminator_optimizer .zero_grad()
        generator_optimizer.zero_grad()

        # Fetch real images and labels
        real_image, label = data
        real_image, label = real_image.to(device), label.to(device)

        # Pass real images through the discriminator
        latent_output, dis_output = discriminator(real_image)

        # Add noise to the latent space
        latent_output_noise = latent_output + opt.epsilon * torch.randn(batch_size, latent_size, device=device)

        # Set the discriminator label for real images
        discriminator_label[:] = real_label_value

        # Compute the discriminator loss for real images
        dis_errD_real = discriminator_criterion(dis_output, discriminator_label)

        if opt.R > 0.0:  # if GAN learning occurs
            (dis_errD_real).backward(retain_graph=True)

        # Compute the KL divergence regularization loss
        kl = kl_loss(latent_output)
        (kl).backward(retain_graph=True)

        # Reconstruct real images from the latent space
        reconstructed_image = generator(latent_output_noise, reverse=False)

        # Compute the reconstruction loss for real images
        rec_real = reconstruction_criterion (reconstructed_image, real_image)

        if opt.W > 0.0:
            (opt.W * rec_real).backward()

        discriminator_optimizer .step()
        generator_optimizer.step()

        # Compute the mean of the discriminator output (between 0 and 1)
        D_x = dis_output.cpu().mean()

        # Compute the norm of the latent space representation
        latent_norm = torch.mean(torch.norm(latent_output.squeeze(), dim=1)).item()


        ###########################
        # NREM perturbed dreaming (N)
        ##########################
        discriminator_optimizer .zero_grad()

        # Detach the latent space representation
        latent_z = latent_output.detach()

        with torch.no_grad():
            # Generate images from the detached latent space
            nrem_image = generator(latent_z)

            # Apply occlusion to the generated images
            occlusion = Occlude(drop_rate=random.random(), tile_size=random.randint(1, 8))
            occluded_nrem_image = occlusion(nrem_image, dim=1)

        # Pass occluded NREM images through the discriminator
        latent_recons_dream, _ = discriminator(occluded_nrem_image)

        # Compute the reconstruction loss for fake images
        rec_fake = reconstruction_criterion (latent_recons_dream, latent_output.detach())

        if opt.N > 0.0:
            (opt.N * rec_fake).backward()

        discriminator_optimizer .step()


       ###########################
        # REM adversarial dreaming (R)
        ##########################

        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        lmbd = opt.lmbd
        noise = torch.randn(batch_size, latent_size, device=device)
        if i==0:
            latent_z = 0.5*latent_output.detach() + 0.5*noise
        else:
            latent_z = 0.25*latent_output.detach() + 0.25*old_latent_output + 0.5*noise
        
        dreamed_image_adv = generator(latent_z, reverse=True) # activate plasticity switch
        latent_recons_dream, dis_output = discriminator(dreamed_image_adv)
        discriminator_label[:] = fake_label_value # should be classified as fake
        dis_errD_fake = discriminator_criterion(dis_output, discriminator_label)
        if opt.R > 0.0: # if GAN learning occurs
            dis_errD_fake.backward(retain_graph=True)
            discriminator_optimizer.step()
            generator_optimizer.step()
        dis_errG = - dis_errD_fake

        D_G_z1 = dis_output.cpu().mean()

        old_latent_output = latent_output.detach()
        
        
        
        ###########################
        # Compute average losses
        ###########################
        store_loss_G.append(dis_errG.item())
        store_loss_D.append((dis_errD_fake + dis_errD_real).item())
        store_loss_R_real.append(rec_real.item())
        store_loss_R_fake.append(rec_fake.item())
        store_norm.append(latent_norm)
        store_kl.append(kl.item())
        


        if i % 200 == 0 and i>1:
            print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f  Loss_R_real: %.4f  Loss_R_fake: %.4f  D(x): %.4f  D(G(z)): %.4f  latent_norm : %.4f  '
                % (epoch, opt.niter, i, len(dataloader),
                    np.mean(store_loss_D), np.mean(store_loss_G), np.mean(store_loss_R_real), np.mean(store_loss_R_fake), D_x, D_G_z1, np.mean(latent_norm) ))
            compare_img_rec = torch.zeros(batch_size * 2, real_image.size(1), real_image.size(2), real_image.size(3))
            with torch.no_grad():
                reconstructed_image = generator(latent_output)
            compare_img_rec[::2] = real_image
            compare_img_rec[1::2] = reconstructed_image
            vutils.save_image(unorm(compare_img_rec[:128]), '%s/recon_%03d.png' % (dir_files, epoch), nrow=8)
            fake = unorm(dreamed_image_adv)
            vutils.save_image(fake[:64].data, '%s/fake_%03d.png' % (dir_files, epoch), nrow=8)
            

    d_losses.append(np.mean(store_loss_D))
    g_losses.append(np.mean(store_loss_G))
    r_losses_real.append(np.mean(store_loss_R_real))
    r_losses_fake.append(np.mean(store_loss_R_fake))
    kl_losses.append(np.mean(store_kl))
    save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses, None, None,  dir_files)

    # do checkpointing
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': generator_optimizer.state_dict(),
        'd_optim': discriminator_optimizer.state_dict(),
        'd_losses': d_losses,
        'g_losses': g_losses,
        'r_losses_real': r_losses_real,
        'r_losses_fake': r_losses_fake,
        'kl_losses': kl_losses,
    }, dir_checkpoint+'/trained.pth')
    
    # save network after 1 learning epoch
    if epoch ==1:
            torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        }, dir_checkpoint+'/trained2.pth')

    print(f'Model successfully saved.')



