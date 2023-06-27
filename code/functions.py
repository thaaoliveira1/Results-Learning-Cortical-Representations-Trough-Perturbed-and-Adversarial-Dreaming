# %%
import torch  # Importing the PyTorch library for deep learning
import numpy as np  # Importing the NumPy library for numerical operations
# plotting
import matplotlib  # Importing the Matplotlib library for data visualization
import torch.utils.data  # Importing PyTorch's utilities for handling datasets
import torchvision.datasets as dset  # Importing the torchvision.datasets module for pre-built datasets
import torchvision.transforms as transforms  # Importing transformations for image preprocessing
from torch.utils.data import Dataset, TensorDataset  # Importing utility classes for custom datasets
from scipy import linalg  # Importing functions from SciPy for linear algebra operations
matplotlib.use('Agg')  # Set the backend of matplotlib to 'Agg', which is a non-interactive backend for saving figures
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib for creating and customizing plots



# %%
# Custom weights initialization called on netG and netD
def initialize_weights(module):
    # Get the name of the module's class
    module_classname = module.__class__.__name__

    # Initialize the weights for convolutional layers
    if module_classname.find('Conv') != -1:
        # Initialize the weights using a normal distribution with mean 0 and standard deviation 0.02
        module.weight.data.normal_(0.0, 0.02)

    # Initialize the weights for batch normalization layers
    elif module_classname.find('BatchNorm') != -1:
        # Initialize the weights using a normal distribution with mean 1.0 and standard deviation 0.02
        module.weight.data.normal_(1.0, 0.02)
        # Fill the bias with zeros
        module.bias.data.fill_(0)


# %% [markdown]
# 1. The code defines a custom weight initialization function called **`weights_init`**. This function is called on the modules **`netG`** and **`netD`** in a network.
# 
# 2. Retrieve the class name of the current module: The function accesses the class name of the current module using **`module.__class__.__name__`**. This allows us to identify the type of the module.
# 
# 3. If the module is a convolutional layer: The code checks if the class name contains the substring 'Conv'. If it does, this indicates that the module is a convolutional layer. In this case, the weights of the module are initialized using a normal distribution with mean 0 and standard deviation 0.02. This initialization helps in randomly initializing the weights close to zero.
# 
# 4. If the module is a batch normalization layer: The code checks if the class name contains the substring 'BatchNorm'. If it does, this indicates that the module is a batch normalization layer. In this case, the weights of the module are initialized using a normal distribution with mean 1.0 and standard deviation 0.02. Additionally, the bias of the module is set to all zeros. This initialization helps in standardizing the inputs during training.
# 
# Overall, this custom weight initialization function ensures that the weights of the convolutional and batch normalization layers are properly initialized with appropriate values, contributing to the effectiveness and stability of the neural network training process.

# %%
import torch
import numpy as np

class Occlude(object):
    def __init__(self, drop_rate=0.0, tile_size=7):
        self.drop_rate = drop_rate
        self.tile_size = tile_size

    def __call__(self, images, dim=0):
        # Create a copy of the input images
        images_modified = images.clone()

        # Determine the device to be used (CPU or GPU)
        if dim == 0:
            device = 'cpu'
        else:
            device = images.get_device()
            if device == -1:
                device = 'cpu'

        # Create a mask tensor of ones with the same size as the images
        mask = torch.ones((images_modified.size(dim), images_modified.size(dim + 1), images_modified.size(dim + 2)),
                          device=device)

        i = 0
        while i < images_modified.size(dim + 1):
            j = 0
            while j < images_modified.size(dim + 2):
                # Randomly drop tiles based on the drop rate
                if np.random.rand() < self.drop_rate:
                    for k in range(mask.size(0)):
                        mask[k, i:i + self.tile_size, j:j + self.tile_size] = 0  # Set the tile to zero in the mask
                j += self.tile_size
            i += self.tile_size

        # Apply the mask to each image by element-wise multiplication
        images_modified = images_modified * mask
        return images_modified


# %% [markdown]
# The **`Occlude`** class represents an image augmentation technique that randomly occludes (drops) tiles within the input images. It can be used to introduce variations during training or testing of a machine learning model.
# 
# 1. The class is initialized with two parameters: **`drop_rate`** (probability of dropping a tile) and **`tile_size`** (size of the tiles to be occluded).
# 
# 2. The **`__call__`** method is defined, which allows the class instance to be called like a function.
# 
# 3. A copy of the input images is created using the **`clone`** method of the **`images`** tensor.
# 
# 4. The device is determined based on the **`dim`** parameter. If **`dim`** is 0, the device is set to CPU; otherwise, it uses the device of the input images tensor.
# 
# 5. A mask tensor is created with the same dimensions as the input images. It is initialized with ones using the **`torch.ones`** function.
# 
# 6. Two nested **`while`** loops are used to iterate over the tiles of the images.
# 
# 7. For each tile, a random number is generated between 0 and 1, and if it is less than the drop rate, the tile is occluded.
# 
# 8. When a tile needs to be occluded, the corresponding region in the mask tensor is set to zero, effectively occluding that tile.

# %%
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_batch):
        # Create a copy of the tensor
        tensor_batch = tensor_batch.clone()

        # Iterate over each channel in the tensor
        for i in range(len(tensor_batch)):
            # Iterate over each channel in the tensor
            for j in range(len(tensor_batch[i])):
                # Unnormalize the tensor by multiplying by the standard deviation and adding the mean
                tensor_batch[i][j].mul_(self.std[j]).add_(self.mean[j])
                # The equivalent of normalization: t.sub_(m).div_(s)

        return tensor_batch

# %% [markdown]
# 1. The code defines a class called **`UnNormalize`** that is used to reverse the normalization applied to a batch of tensor images. Here's how it works:
# 
# 2. The **`__init__`** method initializes the **`UnNormalize`** object with the mean and standard deviation values.
# 
# 3. The **`__call__`** method is invoked when an instance of **`UnNormalize`** is called as a function. It takes a batch of tensor images (**`tensor_batch`**) as input and returns the unnormalized batch of images.
# 
# 4. The method iterates over each tensor in the batch using a **`for`** loop.
# 
# 5. Within the inner loop, it iterates over each channel of the tensor using another **`for`** loop.
# 
# 6. For each channel, it performs the unnormalization by multiplying the tensor values by the corresponding standard deviation (**`self.std[j]`**) and adding the mean value (**`self.mean[j]`**).
# 
# 7. Finally, it returns the unnormalized tensor batch.
# 
# In summary, the **`UnNormalize`** class allows you to reverse the normalization process applied to a batch of tensor images by multiplying the tensor values by the standard deviation and adding the mean for each channel.

# %%
def get_dataset(dataset_name, dataroot, imageSize, is_train=True, drop_rate=0.0, tile_size=32):
    
            # Use the CIFAR10 dataset
    if dataset_name == 'cifar10':
        dataset = dset.CIFAR10(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                # Resize the image to the specified imageSize
                transforms.Resize(imageSize),
                # Convert the image to a tensor
                transforms.ToTensor(),
                # Normalize the image tensor with mean and standard deviation of 0.5
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # Apply Occlude transformation with the specified drop_rate and tile_size
                Occlude(drop_rate=drop_rate, tile_size=tile_size),
            ])
        )
        # Create an instance of the UnNormalize class with mean and standard deviation of 0.5
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # Set the number of image channels to 3 (RGB)
        img_channels = 3
        
            # Use the SVHN dataset
    elif dataset_name == 'svhn':
        if is_train:
            split = 'train'
        else:
            split = 'test'
        dataset = dset.SVHN(
            root=dataroot, download=True,
            split = split,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ]))
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img_channels = 3

            # Use the MNIST dataset
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ])
        )
        unorm = UnNormalize(mean=(0.5,), std=(0.5,))
        img_channels = 1
        
                # Use the FASHION dataset
    elif dataset_name == 'fashion':
        dataset = dset.FashionMNIST(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ]))
                # Create an instance of the UnNormalize class with mean and standard deviation of 0.5
        unorm = UnNormalize(mean=(0.5,), std=(0.5,))
                # Set the number of image channels to 1 (grayscale)
        img_channels = 1
    else:
        raise NotImplementedError("No such dataset {}".format(dataset_name))

    assert dataset
    return dataset, unorm, img_channels

# %% [markdown]
# 1. The code defines a function `get_dataset` that retrieves a dataset for image classification tasks.
# 
# 2. The function takes parameters such as the dataset name, root directory, image size, and optional arguments.
# 
# 3. The function checks the dataset name to determine the appropriate dataset.
# 
# 4. If the dataset is CIFAR-10:
#    a. The CIFAR10 dataset is created with the specified parameters.
#    b. The transformation pipeline includes resizing, tensor conversion, pixel normalization, and occlusion transformation.
#    c. An `UnNormalize` object is created for later visualizations.
#    d. The `img_channels` variable is set to 3.
# 
# 5. If the dataset is SVHN:
#    a. The SVHN dataset is created with the specified parameters.
#    b. The transformation pipeline includes resizing, tensor conversion, pixel normalization, and occlusion transformation.
#    c. An `UnNormalize` object is created for later visualizations.
#    d. The `img_channels` variable is set to 3.
# 
# 6. If the dataset is MNIST:
#    a. The MNIST dataset is created with the specified parameters.
#    b. The transformation pipeline includes resizing, tensor conversion, pixel normalization, and occlusion transformation.
#    c. An `UnNormalize` object is created for later visualizations.
#    d. The `img_channels` variable is set to 1.
# 
# 7. If the dataset is FashionMNIST:
#    a. The FashionMNIST dataset is created with the specified parameters.
#    b. The transformation pipeline includes resizing, tensor conversion, pixel normalization, and occlusion transformation.
#    c. An `UnNormalize` object is created for later visualizations.
#    d. The `img_channels` variable is set to 1.
# 
# 8. If an unsupported dataset name is provided, a `NotImplementedError` is raised.
# 
# 9. The function returns the created dataset object, the unnormalization object, and the number of image channels.

# %%
# Compute the current classification accuracy
def compute_acc(predictions, labels):
    correct = 0
    # Get the predicted labels by selecting the index with the maximum value from the predictions tensor
    predicted_labels = predictions.data.max(1)[1]
    # Compare the predicted labels with the ground truth labels and count the number of correct predictions
    correct = predicted_labels.eq(labels.data).cpu().sum()
    # Compute the accuracy as the percentage of correct predictions
    accuracy = float(correct) / float(len(labels.data)) * 100.0
    return accuracy


def get_latent(dim_latent, batch_size, device):
    # Generate random values from a normal distribution with mean 0 and standard deviation 1
    latent_z = np.random.normal(0, 1, (batch_size, dim_latent))
    # Convert the generated values to a tensor and move it to the specified device (e.g., CPU or GPU)
    latent_z = torch.tensor(latent_z, dtype=torch.float32, device=device)
    # Reshape the tensor to have dimensions (batch_size, dim_latent, 1, 1)
    latent_z = latent_z.view(batch_size, dim_latent, 1, 1)
    return latent_z

# %% [markdown]
# 1. The **`compute_acc`** function takes in two arguments: **`predictions`** and **`labels`**, which represent the predicted values and the ground truth labels, respectively.
# 
# 2. The **`correct`** variable is initialized to 0.
# 
# 3. **`preds_`** is assigned the predicted labels by selecting the index with the maximum value from the **`predictions`** tensor.
# 
# 4. The **`eq`** function compares the predicted labels with the ground truth labels, and the **`cpu`** function moves the tensor to the CPU memory if it was on a GPU.
# 
# 5. The number of correct predictions is computed by summing up the values where the predicted label equals the ground truth label.
# 
# 6. The accuracy is calculated by dividing the number of correct predictions by the total number of labels and multiplying by 100 to get the percentage.
# 
# 7. The **`get_latent`** function takes three arguments: **`dim_latent`**, **`batch_size`**, and **`device`**. It is used to generate a tensor of random values from a normal distribution.
# 
# 8. Random values are generated using **`np.random.normal`** with a mean of 0 and a standard deviation of 1, resulting in a **`(batch_size, dim_latent)`** array.
# 
# 9. The generated values are then converted to a PyTorch tensor of type **`torch.float32`** and moved to the specified **`device`**.
# 
# 10. The tensor is reshaped to have dimensions **`(batch_size, dim_latent, 1, 1)`** to match the expected input shape for further processing.

# %%
def save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses, fid_nrem, fid_rem,  files_dir):
    # Create an array of epoch values from 0 to 'epoch'
    epochs = np.arange(0, epoch+1)
    # Create a new figure with a size of 10x5 inches
    fig = plt.figure(figsize=(10, 5))
    # Add the first subplot (1 row, 2 columns, subplot 1)
    ax1 = fig.add_subplot(121)
    
    # Plot the generator losses if available
    if g_losses is not None:
        ax1.plot(epochs, g_losses, label='generator (REM)')
    
    # Plot the discriminator losses if available
    if d_losses is not None:
        ax1.plot(epochs, d_losses, color='green', label='discriminator (Wake, REM)')
    
    # Set the x-axis label
    ax1.set_xlabel('epochs')
    # Set the y-axis label
    ax1.set_ylabel('loss')
    # Set the title of the subplot
    ax1.set_title('losses with training')
    
    # Plot the real data reconstruction losses if available
    if r_losses_real is not None:
        ax1.plot(epochs, r_losses_real, color='orange', label='data rec. (Wake)')
    
    # Plot the fake data reconstruction losses if available
    if r_losses_fake is not None:
        ax1.plot(epochs, r_losses_fake, color='magenta', label='latent rec. (NREM)')
    
    # Plot the KL divergence losses if available
    if kl_losses is not None:
        ax1.plot(epochs, kl_losses, color='brown', label='KL div. (Wake)')
    
    # Add a legend to the subplot
    ax1.legend()
    
    # Check if FID scores for NREM and REM are available
    if fid_nrem is not None and fid_rem is not None:
        # Add the second subplot (1 row, 2 columns, subplot 2)
        ax2 = fig.add_subplot(122)
        # Plot the FID scores for NREM and REM
        ax2.plot(epochs, fid_nrem, color='darkorange', label='FID NREM')
        ax2.plot(epochs, fid_rem, color='magenta', label='FID REM')
        # Add a legend to the subplot
        ax2.legend()
    
    # Save the figure as 'losses.pdf' in the specified directory
    fig.savefig(files_dir + '/losses.pdf')


# %% [markdown]
# The code defines a function called **`save_loss_fig`** that is responsible for creating a figure and saving it as a PDF file, depicting the losses and FID (Fréchet Inception Distance) scores during training. Here's how it works:
# 
# 1. The function takes several input parameters: **`epoch`** (the current epoch), various loss values (**`generator_losses`**, **`discriminator_losses`**, **`real_losses_real`**, **`real_losses_fake`**, **`kl_losses`**), FID scores (**`fid_nrem`**, **`fid_rem`**), and the output directory path (**`output_dir`**).
# 
# 2. An array of epoch values is created using **`np.arange`** to span from 0 to the current epoch.
# 
# 3. A figure is created using **`plt.figure(figsize=(10, 5))`** with a size of 10x5 inches.
# 
# 4. The first subplot (ax1) is added to the figure (121 denotes a grid of 1 row and 2 columns, with this subplot occupying the first position).
# 
# 5. If generator losses are provided (**`generator_losses`** is not **`None`**), they are plotted against the epochs with the label 'generator (REM)'.
# 
# 6. If discriminator losses are provided (**`discriminator_losses`** is not **`None`**), they are plotted in green color with the label 'discriminator (Wake, REM)'.
# 
# 7. If real losses for real data (**`real_losses_real`**) are provided, they are plotted in orange color with the label 'data rec. (Wake)'.
# 
# 8. If real losses for fake data (**`real_losses_fake`**) are provided, they are plotted in magenta color with the label 'latent rec.

# %%
import numpy as np
import matplotlib.pyplot as plt

def save_fig_trainval(epoch, losses, accuracies, directory):
    # Generate the x-axis values from 0 to epoch
    epochs = np.arange(0, epoch+1)

    # Create a new figure with a size of 10x5 inches
    fig = plt.figure(figsize=(10, 5))

    # Add the first subplot for loss values
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, losses['train'], label='train loss')  # Plot train loss
    ax1.plot(epochs, losses['val'], label='validation loss')  # Plot validation loss
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    # Add the second subplot for accuracy values
    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, accuracies['train'], label='train accuracy')  # Plot train accuracy
    ax2.plot(epochs, accuracies['val'], label='val accuracy')  # Plot validation accuracy
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_ylim(0, 100)  # Set the y-axis limits
    ax2.legend()

    # Save the figure as a PDF file in the specified directory
    fig.savefig(directory + '/trainval.pdf')


# %% [markdown]
# The **`save_fig_trainval`** function takes several inputs: **`epoch`** (the number of epochs), **`losses`** (a dictionary containing loss values for training and validation), **`accuracies`** (a dictionary containing accuracy values for training and validation), and **`directory`** (the directory path where the resulting PDF file will be saved).
# 
# 1. The **`epochs`** variable is created using NumPy's **`arange`** function to generate values from 0 to **`epoch+1`**.
# 
# 2. A new figure object **`fig`** is created with a size of 10x5 inches using **`plt.figure(figsize=(10, 5))`**.
# 
# 3. The first subplot **`ax1`** is added to the figure at position 121 (1 row, 2 columns, 1st position).
# 
# 4. The **`ax1.plot`** function is called twice to plot the training and validation losses against the epochs using the values from the **`losses`** dictionary.
# 
# 5. The x-label and y-label for the first subplot are set using **`ax1.set_xlabel`** and **`ax1.set_ylabel`**.
# 
# 6. A legend is added to the first subplot using **`ax1.legend`**.
# 
# 7. The second subplot **`ax2`** is added to the figure at position 122 (1 row, 2 columns, 2nd position).
# 
# 8. The **`ax2.plot`** function is called twice to plot the training and validation accuracies against the epochs using the values from the **`accuracies`** dictionary.
# 
# 9. The x-label and y-label for the second subplot are set using **`ax2.set_xlabel`** and **`ax2.set_ylabel`**.
# 
# 10. The y-axis limits for the second subplot are set to range from 0 to 100 using **`ax2.set_ylim`**.
# 
# 11. A legend is added to the second subplot using **`ax2.legend`**.
# 
# 12. Finally, the figure is saved as a PDF file in the specified **`directory`** using **`fig.savefig`**.
# 
# This function is useful for visualizing the training and validation progress of a machine learning model. It generates a single figure with two subplots: one for plotting the loss values and another for plotting the accuracy values. The resulting figure is saved as a PDF file for further analysis or reporting

# %%
import torch
from scipy import stats

def kl_loss(latent_output):
    # Compute the mean and standard deviation along the batch dimension
    mean = torch.mean(latent_output, dim=0)
    std = torch.std(latent_output, dim=0)
    
    # Compute the KL divergence loss using the formula
    kl_loss = torch.mean((std ** 2 + mean ** 2) / 2 - torch.log(std) - 1/2)
    return kl_loss


def calculate_activation_statistics(images, model, batch_size=128, dims=2048, cuda=False):
    model.eval()
    act = np.empty((len(images), dims))
    
    # Move images to GPU if cuda is enabled
    if cuda:
        batch = images.cuda()
    else:
        batch = images
    
    # Get the model predictions for the batch of images
    pred = model(batch)[0]
    
    # If the model output is not a scalar, apply global spatial average pooling
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    
    # Convert the tensor to numpy array and reshape it
    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    return act
    
    # Compute the mean and covariance of the activation values
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def mean_and_sem(array, color=None, axis=0):
    # Compute the mean of the array along the specified axis
    mean = array.mean(axis=0)
    
    # Compute the standard error of the mean (SEM)
    sem_plus = mean + stats.sem(array, axis=axis)
    sem_minus = mean - stats.sem(array, axis=axis)
    
    # Fill the area between the upper and lower SEM with the specified color
    if color is not None:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, color=color, alpha=0.5)
    else:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    
    return mean


# %% [markdown]
# 1. The **`kl_loss`** function calculates the Kullback-Leibler (KL) divergence loss for a given **`latent_output`**. It computes the mean and standard deviation of the **`latent_output`**, and then uses these values to compute the KL divergence loss according to the formula.
# 
# 2. The **`calculate_activation_statistics`** function is used to compute the mean and covariance of the activation values produced by a given **`model`** for a batch of **`images`**. It first sets the model to evaluation mode and initializes an array **`act`** to store the activation values. It then moves the images to the GPU if **`cuda`** is enabled. The model is applied to the batch of images, and if the model output is not a scalar, global spatial average pooling is performed. The resulting tensor is converted to a numpy array and reshaped before returning it.
# 
# 3. The **`mean_and_sem`** function calculates the mean and standard error of the mean (SEM) for a given array **`array`** along the specified **`axis`**. It computes the mean along the axis and then calculates the upper and lower SEM by adding and subtracting the SEM, respectively, using the **`stats.sem`** function from SciPy. Finally, it fills the area between the upper and lower SEM with a specified color.
# 
# Overall, this code provides utility functions for calculating KL divergence loss, activation statistics, and mean with SEM, which can be useful in various machine learning and statistical analysis tasks.

# %%
from scipy.linalg import sqrtm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Numpy implementation of the Frechet Distance.
    # The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    # and X_2 ~ N(mu_2, C_2) is
    #         d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    
    # Convert mu1 and mu2 to arrays with at least 1 dimension
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    # Convert sigma1 and sigma2 to arrays with at least 2 dimensions
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Check if mu1 and mu2 have the same shape
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    
    # Check if sigma1 and sigma2 have the same shape
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    # Calculate the difference between mu1 and mu2
    diff = mu1 - mu2

    # Calculate the square root of the matrix product of sigma1 and sigma2
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check if the covmean contains any non-finite values
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        # Add epsilon to the diagonal of sigma1 and sigma2
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Check if the covmean is a complex object
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        # Take the real component of the covmean
        covmean = covmean.real

    # Calculate the trace of covmean
    tr_covmean = np.trace(covmean)

    # Calculate and return the Frechet distance
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_frechet(inception_real, inception_fake, model, return_statistics=False):
    # Calculate the mean and covariance of the real and fake inception outputs
    mu_1 = np.mean(inception_real, axis=0)
    mu_2 = np.mean(inception_fake, axis=0)
    std_1 = np.cov(inception_real, rowvar=False)
    std_2 = np.cov(inception_fake, rowvar=False)
    
    # Calculate the Frechet distance
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)

    return fid_value


# %% [markdown]
# 1. **`calculate_frechet_distance`** is a function that calculates the Frechet distance between two multivariate Gaussian distributions. It takes four parameters: **`mu1`**, **`sigma1`**, **`mu2`**, and **`sigma2`**, which represent the mean and covariance matrices of the two distributions. The optional parameter **`eps`** is used for numerical stability. The function implements the formula for the Frechet distance: **`d^2 = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))`**, where **`mu1`** and **`mu2`** are the means, **`C1`** and **`C2`** are the covariance matrices, and **`Tr`** denotes the trace operation.
# 
# 2. The function first ensures that **`mu1`** and **`mu2`** are at least 1-dimensional arrays, and **`sigma1`** and **`sigma2`** are at least 2-dimensional arrays using the **`np.atleast_1d`** and **`np.atleast_2d`** functions.
# 
# 3. It then checks if **`mu1`** and **`mu2`** have the same shape, and if **`sigma1`** and **`sigma2`** have the same shape. If the shapes are not equal, it raises assertions to indicate that the mean vectors and covariance matrices have different lengths or dimensions.
# 
# 4. The function calculates the difference between **`mu1`** and **`mu2`** and stores it in the variable **`diff`**.
# 
# 5. Next, it computes the square root of the matrix product of **`sigma1`** and **`sigma2`** using the **`sqrtm`** function from **`scipy.linalg`**. This operation is stored in the variable **`covmean`**.
# 
# 6. The function checks if **`covmean`** contains any non-finite (e.g., NaN or infinity) values. If it does, it adds a small epsilon value to the diagonal of **`sigma1`** and **`sigma2`** to ensure numerical stability. This step helps handle cases where the covariance matrices are ill-conditioned or singular.
# 
# 7. If the **`covmean`** matrix is complex, the function checks if the imaginary components on the diagonal are close to zero within a tolerance. If not, it raises a **`ValueError`** indicating the presence of a significant imaginary component.
# 
# 8. Finally, the function calculates the trace of **`covmean`** and returns the Frechet distance as the sum of the squared difference between the means and the trace of the covariance matrices.
# 
# 9. The **`calculate_frechet`** function takes two inputs: **`inception_real`** and **`inception_fake`**, which represent the real and fake samples, respectively, and calculates the Frechet Inception Distance (FID) between them. It also takes a **`model`** parameter, which is not used in the provided code snippet.
# 
# 10. Inside **`calculate_frechet`**, the mean (**`mu_1`** and **`mu_2`**) and covariance (**`std_1`** and **`std_2`**) of the real and fake samples are calculated using the **`np.mean`** and **`np.cov`** functions.
# 
# 11. The FID value is obtained by calling the **`calculate_frechet_distance`** function with the calculated mean and covariance values.
# 
# 12. Finally, the FID value is returned as the result of the **`calculate_frechet`** function.
# 
# In summary, the code provides a way to calculate the Frechet Inception Distance, which is a measure of similarity between two sets of samples based on their mean vectors and covariance matrices. The **`calculate_frechet_distance`** function implements the mathematical formula for the Frechet distance, while the **`calculate_frechet`** function orchestrates the calculation by extracting the mean and covariance values from the input samples and calling the `calculate_frechet_distance

# %% [markdown]
# 


