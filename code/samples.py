# %%
from __future__ import print_function  # Enable compatibility with Python 2 and 3 print statements
import argparse  # Module for parsing command-line arguments
import os  # Module for interacting with the operating system
import numpy as np  # Numerical computing library
import torch.optim as optim  # Optimization algorithms for PyTorch
import torch.utils.data  # Utilities for handling data in PyTorch
import torchvision.utils as vutils  # Utility functions for image visualization in PyTorch
from torch.autograd import Variable  # Functionality for automatic differentiation in PyTorch
from functions import *  # Import functions from a custom functions module
from model import *  # Import classes and functions from a custom model module


# %%
# Importing the necessary libraries
import argparse

# Creating an argument parser
parser = argparse.ArgumentParser()

# Adding arguments with their default values and descriptions
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--dataset', default='mnist', help='Dataset to use: cifar10 | imagenet | mnist')
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
parser.add_argument('--output_folder', default='trained_mnist', help='Folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)



# Parsing the command-line arguments
opt, unknown = parser.parse_known_args()

# Set the number of iterations to the number of epochs
opt.niter = opt.num_epochs

# Assign the value of latent_size based on opt.latent_size
latent_size = opt.latent_size

# Printing the parsed arguments
print(opt)


# %%
# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

dir_files = './results/'+opt.dataset+'/'+opt.outf
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf
try:
    os.makedirs(dir_files)
except OSError:
    pass
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), drop_last=True)



# %%
# Define and assign values to hyperparameters
num_gpus = int(opt.num_gpus)
latent_dim = int(opt.latent_size)
batch_size = opt.batch_size
lmbd = 0.5


# Instantiate generator and discriminator networks
generator = Generator(num_gpus, latent_dim=latent_dim, ngf=opt.num_filters, img_channels=img_channels)
generator.apply(initialize_weights)
discriminator = Discriminator(num_gpus, latent_dim=latent_dim, ndf=opt.num_filters, img_channels=img_channels, dropout_prob=opt.dropout_prob)
discriminator.apply(initialize_weights)

# Move networks to the GPU
generator.to(device)
discriminator.to(device)

# %%
if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# %% [markdown]
# 1. The code first checks if a trained model checkpoint file exists using the **`os.path.exists()`** function.
# 
# 2. If a checkpoint file exists, it indicates that a pre-trained model is available, and the code proceeds to load the data from the checkpoint.
# 
# 3. The **`torch.load()`** function is used to load the checkpoint file, specifying the path to the file (**`dir_checkpoint+'/trained.pth'`**) and the **`map_location`** parameter as **`'cpu'`** to ensure compatibility.
# 
# 4. The generator and discriminator models (**`netG`** and **`netD`**) are updated with the saved state dictionaries using the **`load_state_dict()`** method.
# 
# 5. A message is printed indicating that training will start from the loaded model.
# 
# 6. If no trained model checkpoint file exists, the code proceeds to the **`else`** block and prints a message indicating that training will restart.
# 
# In summary, this code checks if a pre-trained model checkpoint file exists and loads the model from the checkpoint if it exists. If no checkpoint file is found, it assumes that training will start from scratch.

# %%
# load epoch 3 networks
netG3 = Generator(num_gpus, latent_dim=latent_size, img_channels=img_channels)
netG3.apply(initialize_weights)
netD3 = Discriminator(num_gpus, latent_dim=latent_size, img_channels=img_channels)
netD3.apply(initialize_weights)
# send to GPU
netD3.to(device)
netG3.to(device)


# %%
# Check if the checkpoint file exists
if os.path.exists(dir_checkpoint+'/trained2.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    # Load the checkpoint data
    checkpoint = torch.load(dir_checkpoint+'/trained2.pth', map_location='cpu')
    # Load the generator state_dict from the checkpoint
    netG3.load_state_dict(checkpoint['generator'])
    # Load the discriminator state_dict from the checkpoint
    netD3.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')


# %% [markdown]
# 1. The code checks if a specific checkpoint file (**`trained2.pth`**) exists in the specified directory.
# 
# 2. If the checkpoint file exists, it proceeds to load the pre-trained model.
# 
# 3. It prints a message indicating that the pre-trained model is being loaded.
# 
# 4. The **`torch.load()`** function is used to load the checkpoint data into a dictionary.
# 
# 5. The generator's state_dict is loaded from the checkpoint dictionary into the **`netG3`** model.
# 
# 6. The discriminator's state_dict is loaded from the checkpoint dictionary into the **`netD3`** model.
# 
# 7. A message is printed to indicate that training will start from the loaded model if it exists.
# 
# 8. If the checkpoint file does not exist, a message is printed indicating that no pre-trained model is detected, and training should be restarted.
# 
# In summary, this code checks if a specific checkpoint file exists and loads the generator and discriminator models' state_dicts from the checkpoint file if it exists. It allows you to continue training from a saved model checkpoint or start training from scratch if no checkpoint is found.

# %%
# Prepare images
dataloader_iter = iter(dataloader)
# Get the first batch of images and their corresponding labels
image_eval1, _ = next(dataloader_iter)
# Get the second batch of images and their corresponding labels
image_eval2, _ = next(dataloader_iter)
# Create clones of the image batches for saving
image_eval1_save = image_eval1.clone()
image_eval2_save = image_eval2.clone()
# Save the first 3 images from the first batch as evaluation wake1 images
vutils.save_image(unorm(image_eval1_save[:3]).data, '%s/eval_wake1_%03d.png' % (dir_files, 0), nrow=1)
# Save the first 3 images from the second batch as evaluation wake2 images
vutils.save_image(unorm(image_eval2_save[:3]).data, '%s/eval_wake2_%03d.png' % (dir_files, 0), nrow=1)


# %% [markdown]
# 1. The code prepares the images for evaluation by obtaining two batches of images from the dataloader.
# 
# 2. It creates an iterator (**`dataloader_iter`**) from the dataloader to iterate over the batches.
# 
# 3. The **`next()`** function is used to get the next batch of images and their corresponding labels from the iterator. Here, **`image_eval1`** and **`_`** are used to store the images and labels of the first batch, respectively.
# 
# 4. Similarly, the **`next()`** function is used again to get the next batch of images and their corresponding labels. Here, **`image_eval2`** and **`_`** are used to store the images and labels of the second batch.
# 
# 5. Clones of the image batches are created for saving purposes. This is done to preserve the original images before any transformations.
# 
# 6. The **`vutils.save_image()`** function is used to save the first 3 images from the first batch (**`image_eval1_save`**) as evaluation wake1 images. The **`unorm()`** function is used to unnormalize the image tensor before saving.
# 
# 7. Similarly, the **`vutils.save_image()`** function is used to save the first 3 images from the second batch (**`image_eval2_save`**) as evaluation wake2 images. Again, the **`unorm()`** function is used to unnormalize the image tensor before saving.
# 
# In summary, this code prepares two batches of images for evaluation. It saves the first 3 images from each batch as wake1 and wake2 evaluation images, respectively, by unnormalizing and saving the image tensors using the **`vutils.save_image()`** function.

# %%
# Generate samples with final epoch networks
with torch.no_grad():
    # Move the images to the device (GPU)
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    # Get the latent outputs from the discriminator for the images
    latent_output1, _ = discriminator(image_eval1)
    latent_output2, _ = discriminator(image_eval2)
    # Generate NREM (Non-REM) sample using the generator
    nrem = generator(latent_output1)
    # Generate random noise vector
    noise = torch.randn(batch_size, latent_size, device=device)
    # Combine the latent outputs, noise, and perform linear interpolation
    latent_rem = 0.25 * latent_output1 + 0.25 * latent_output2 + 0.5 * noise
    # Generate REM (Rapid Eye Movement) sample using the generator
    rem = generator(latent_rem)

# Unnormalize the generated samples
nrem = unorm(nrem)
rem = unorm(rem)
rec_image_eval1 = unorm(rec_image_eval1)
rec_image_eval2 = unorm(rec_image_eval2)

# Save the generated samples
vutils.save_image(rec_image_eval1[:3].data, '%s/eval_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval_rec2.png' % (dir_files), nrow=1)
vutils.save_image(nrem[:3].data, '%s/eval_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval_rem.png' % (dir_files), nrow=1)


# %% [markdown]
# 1. The code generates samples using the final epoch networks.
# 
# 2. The **`torch.no_grad()`** context manager is used to disable gradient calculations, as we are only generating samples and not training.
# 
# 3. The images (**`image_eval1`** and **`image_eval2`**) are moved to the device (GPU) using the **`to()`** method.
# 
# 4. The discriminator is used to obtain the latent outputs (**`latent_output1`** and **`latent_output2`**) for the images.
# 
# 5. The generator is used to generate the NREM (Non-REM) sample (**`nrem`**) by passing the **`latent_output1`** through it.
# 
# 6. Random noise is generated using **`torch.randn()`**.
# 
# 7. The latent outputs, noise, and original images are combined to create a blended latent representation for the REM (Rapid Eye Movement) sample (**`latent_rem`**).
# 
# 8. The generator is then used to generate the REM sample (**`rem`**) by passing the **`latent_rem`** through it.
# 
# 9. The generated samples are unnormalized using the **`unorm()`** function.
# 
# 10. The generated samples and reconstructed images are saved using the **`vutils.save_image()`** function.
# 
# In summary, this code generates samples using the final epoch networks. It uses the discriminator to obtain latent outputs for the input images and generates NREM and REM samples using the generator. The generated samples are then unnormalized and saved as images.

# %%
# Generate samples with epoch 3 networks
with torch.no_grad():
    # Move the images to the device (GPU)
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    
    # Get the latent outputs from the discriminator for the images
    latent_output1, _ = netD3(image_eval1)
    latent_output2, _ = netD3(image_eval2)
    
    # Generate NREM (Non-REM) sample using the generator
    nrem = netG3(latent_output1)
    
    # Generate random noise vector
    noise = torch.randn(batch_size, latent_size, device=device)
    
    # Combine the latent outputs, noise, and perform linear interpolation
    latent_rem = 0.25 * latent_output1 + 0.25 * latent_output2 + 0.5 * noise
    
    # Generate REM (Rapid Eye Movement) sample using the generator
    rem = netG3(latent_rem)

# Unnormalize the generated samples
nrem = unorm(nrem)
rem = unorm(rem)
rec_image_eval1 = unorm(rec_image_eval1)
rec_image_eval2 = unorm(rec_image_eval2)

# Save the generated samples
vutils.save_image(rec_image_eval1[:3].data, '%s/eval3_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval3_rec2.png' % (dir_files), nrow=1)
vutils.save_image(nrem[:3].data, '%s/eval3_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval3_rem.png' % (dir_files), nrow=1)


# %% [markdown]
# 1. The code is generating samples using the networks trained until epoch 3.
# 
# 2. The **`torch.no_grad()`** context manager is used to disable gradient calculation, reducing memory usage and speeding up computations.
# 
# 3. The images **`image_eval1`** and **`image_eval2`** are moved to the device (GPU).
# 
# 4. The latent outputs of the discriminator for the images are obtained.
# 
# 5. The generator **`netG3`** generates NREM samples using the first latent output.
# 
# 6. Random noise is generated using **`torch.randn`**.
# 
# 7. The latent outputs, noise, and original latent outputs are combined and interpolated to create the latent input for generating REM samples.
# 
# 8. The generator **`netG3`** generates REM samples using the interpolated latent input.
# 
# 9. The generated samples are unnormalized using the **`unorm`** function.
# 
# 10. The reconstructed images (**`rec_image_eval1`** and **`rec_image_eval2`**) are unnormalized.
# 
# 11. The generated samples and reconstructed images are saved as images.
# 
# Please note that the variables **`rec_image_eval1`** and **`rec_image_eval2`** should be defined and assigned values before this section of the code is executed.

# %% [markdown]
# 


