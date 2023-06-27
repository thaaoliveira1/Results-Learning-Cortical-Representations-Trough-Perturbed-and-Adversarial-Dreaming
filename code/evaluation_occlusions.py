# %%
from __future__ import print_function  # Importing a print function from future versions of Python
import argparse  # Importing the argparse module for command-line argument parsing
import os  # Importing the os module for operating system related functionalities
import numpy as np  # Importing the numpy library for numerical computations
import torch.optim as optim  # Importing the optim module from the torch library for optimization algorithms
import torch.utils.data  # Importing the data module from the torch library for handling data loading and processing
import torchvision.utils as vutils  # Importing the vutils module from torchvision for image utilities
from torch.autograd import Variable  # Importing the Variable class from torch.autograd for automatic differentiation
from functions import *  # Importing custom functions from a module named 'functions'
from model import *  # Importing custom models from a module named 'model'


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fashion', help='cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--niterC', type=int, default=20, help='number of epochs to train the classifier')
parser.add_argument('--nf', type=int, default=64, help='filters factor')
parser.add_argument('--drop', type=float, default=0, help='probably of dropping a patch')
parser.add_argument('--lrC', type=float, default=0.2, help='learning rate of the classifier, default=0.0002')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--acc_file', default='accuracies_occ.pth', help='folder to output accuracies')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--tile_size', type=int, default=4, help='tile size for occlusions')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')

opt, unknown = parser.parse_known_args()
print(opt)


# %%
# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

dir_files = './results/'+opt.dataset+'/'+opt.outf
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf
acc_file = opt.acc_file

try:
    os.makedirs(dir_files)
except OSError:
    pass
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

drop_rate = opt.drop/100.0


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if opt.dataset == 'cifar10':
    n_train = 50000
    n_test = 10000
elif opt.dataset == 'svhn':
    n_train = 58606
    n_test = 14651
elif opt.dataset == 'mnist':
    n_train = 50000
    n_test = 10000
elif opt.dataset == 'fashion':
    n_train = 50000
    n_test = 10000



# %%
# train dataset with occlusions (param drop_rate)
dataset, unorm, img_channels = get_dataset(dataset_name=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize, is_train=True, drop_rate=drop_rate, tile_size=opt.tile_size)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_train, shuffle=True, num_workers=int(opt.workers), drop_last=True)
# test dataset with occlusions (param drop_rate)
test_dataset, unorm, img_channels = get_dataset(dataset_name=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize, is_train=False, drop_rate=drop_rate, tile_size=opt.tile_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test, shuffle=False, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)
batch_size = opt.batchSize

netG = Generator(ngpu, latent_dim=nz, ngf=opt.nf, img_channels=img_channels)
netG.apply(initialize_weights)
netD = Discriminator(ngpu, latent_dim=nz, ndf=opt.nf, img_channels=img_channels,  dropout_prob=opt.drop)
netD.apply(initialize_weights)
# send to GPU
netD.to(device)
netG.to(device)

# %% [markdown]
# - The code snippet initializes the training and test datasets with occlusions using the **`get_dataset`** function. It passes various parameters such as the dataset name, data root, image size, training flag, drop rate, and tile size.
# - The **`get_dataset`** function returns the dataset, a normalization function (**`unorm`**), and the number of image channels. These values are assigned to the variables **`dataset`**, **`unorm`**, and **`img_channels`**, respectively.
# - Data loaders are then created for the training and test datasets using the **`torch.utils.data.DataLoader`** class. These data loaders handle the loading of data in batches during training and testing.
# - Several hyperparameters such as the number of GPUs to use (**`ngpu`**), size of the input latent vector (**`nz`**), number of classes for classification (**`num_classes`**), and batch size (**`batch_size`**) are defined.
# - Instances of the Generator and Discriminator models are created (**`netG`** and **`netD`**, respectively). The parameters passed to these models include the number of GPUs, latent dimension, number of generator filters (**`ngf`**), number of discriminator filters (**`ndf`**), image channels, and dropout probability (**`dropout_prob`**).
# - The **`apply`** method is used to apply weight initialization (**`initialize_weights`**) to the Generator and Discriminator models.
# - Finally, the Generator and Discriminator models are sent to the specified device (GPU) using the **`to`** method.
# 
# Overall, this code sets up the necessary components for training and testing a generative adversarial network (GAN). It initializes the datasets, creates data loaders, defines model hyperparameters, creates instances of the Generator and Discriminator models, applies weight initialization, and sends the models to the GPU for accelerated computation.

# %%
if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    d_losses = checkpoint.get('d_losses', [float('inf')])
    g_losses = checkpoint.get('g_losses', [float('inf')])
    r_losses = checkpoint.get('r_losses', [float('inf')])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# %%
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

if os.path.exists(dir_files+'/' + acc_file):
    # Load data from last checkpoint
    print('Loading accuracies...')
    checkpoint = torch.load(dir_files+'/'+acc_file, map_location='cpu')
    train_accuracies = checkpoint.get('train_accuracies', [float('inf')])
    test_accuracies = checkpoint.get('test_accuracies', [float('inf')])
    train_losses = checkpoint.get('train_losses', [float('inf')])
    test_losses = checkpoint.get('test_losses', [float('inf')])
else:
    print('No accuracies found...')

# %%
n_epochs_c = opt.niterC
class_criterion = nn.CrossEntropyLoss()

print("Storing training representations ...")
image, label = next(iter(train_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    train_features = latent_output.cpu()
    train_labels = label.cpu().long()

print("Storing validation representations ...")
image, label = next(iter(test_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    test_features = latent_output.cpu()
    test_labels = label.cpu().long()


# %%
# create dataset of latent activities
linear_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
linear_test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

linear_train_loader = torch.utils.data.DataLoader(linear_train_dataset, batch_size=batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
linear_test_loader = torch.utils.data.DataLoader(linear_test_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)

classifier = OutputClassifier(nz, num_classes=num_classes)
classifier.to(device)
optimizerC = optim.SGD(classifier.parameters(), lr=opt.lrC)

if os.path.exists(dir_checkpoint + '/trained_classifier.pth'):
    # Load data from last checkpoint
    print('Loading trained classifier...')
    checkpoint = torch.load(dir_checkpoint + '/trained_classifier.pth', map_location='cpu')
    classifier.load_state_dict(checkpoint['classifier'])
    print('Use trained classifier...')
else:
    print('No trained classifier detecte...')

# %%
# Initialize lists to store training and testing metrics
store_train_acc = []
store_test_acc = []
store_train_loss = []
store_test_loss = []

print("training on train set...")

# Iterate over batches of data in the training set
for feature, label in linear_train_loader:
    feature, label = feature.to(device), label.to(device)
    classifier.eval()  # Set the classifier to evaluation mode
    class_output = classifier(feature)  # Forward pass: compute the output of the classifier
    class_err = class_criterion(class_output, label)  # Compute the classification error
    # Store training metrics
    train_acc = compute_acc(class_output, label)  # Compute the training accuracy
    store_train_acc.append(train_acc)  # Append the training accuracy to the list
    store_train_loss.append(class_err.item())  # Append the training loss to the list

print("testing on test set...")
# Iterate over batches of data in the testing set
for feature, label in linear_test_loader:
    feature, label = feature.to(device), label.to(device)
    classifier.eval()  # Set the classifier to evaluation mode
    class_output = classifier(feature)  # Forward pass: compute the output of the classifier
    class_err = class_criterion(class_output, label)  # Compute the classification error
    # Store testing metrics
    test_acc = compute_acc(class_output, label)  # Compute the testing accuracy
    store_test_acc.append(test_acc)  # Append the testing accuracy to the list
    store_test_loss.append(class_err.item())  # Append the testing loss to the list

# Print the average training and testing metrics for the current epoch
print('[%d/%d]  train_loss: %.4f  test_loss: %.4f  train_acc: %.4f  test_acc: %.4f'
      % (opt.niterC, n_epochs_c, np.mean(store_train_loss), np.mean(store_test_loss),
         np.mean(store_train_acc), np.mean(store_test_acc)))


# %% [markdown]
# - The code initializes empty lists to store the training and testing metrics, including training accuracy, testing accuracy, training loss, and testing loss.
# - It then enters a loop to train the linear classifier for each epoch (**`opt.niterC`** is the total number of epochs).
# - Inside the loop, it first performs training on the training set.
#     - It iterates over batches of data in the **`linear_train_loader`**.
#     - The input features and labels are moved to the specified device (GPU) if available.
#     - The classifier is set to evaluation mode using **`classifier.eval()`** to ensure that any layers like dropout behave correctly during evaluation.
#     - The forward pass is performed by passing the features through the classifier (**`class_output = classifier(feature)`**).
#     - The classification error is computed using the specified loss criterion (**`class_err = class_criterion(class_output, label)`**).
#     - The training accuracy is calculated by calling the **`compute_acc`** function, which compares the predicted labels with the ground truth labels.
#     - The training accuracy and loss are then appended to their respective lists.
# - After training on the training set, the code proceeds to evaluate the model on the testing set.
#     - It follows a similar procedure as the training loop, iterating over batches of data in the **`linear_test_loader`** and computing the testing accuracy and loss.
# - Finally, the code prints the average training and testing metrics for the current epoch, including the training loss, testing loss, training accuracy, and testing accuracy.
# 
# The code essentially trains a linear classifier using the features extracted from the latent vectors obtained from a GAN. It performs forward passes on both the training and testing datasets, calculates the classification errors, and stores the accuracy and loss metrics for analysis. This allows monitoring the classifier's performance and evaluating how well it generalizes to unseen data during the training process.

# %%
# Append the average training metrics to the respective lists
train_accuracies.append(np.mean(store_train_acc))
train_losses.append(np.mean(store_train_loss))

# Append the average testing metrics to the respective lists
test_accuracies.append(np.mean(store_test_acc))
test_losses.append(np.mean(store_test_loss))

# Save the average metrics to a file
torch.save({
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'train_losses': train_losses,
    'test_losses': test_losses,
}, dir_files + '/' + acc_file)
print(f'Accuracies successfully saved.')

# Create a figure for plotting
e = np.arange(0, len(train_accuracies))
fig = plt.figure(figsize=(10, 5))

# Add subplot for loss plot
ax1 = fig.add_subplot(121)
ax1.plot(e, train_losses, label='train loss')
ax1.plot(e, test_losses, label='test loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.set_title('losses with uns. training')
ax1.legend()

# Add subplot for accuracy plot
ax2 = fig.add_subplot(122)
ax2.plot(e, train_accuracies, label='train acc')
ax2.plot(e, test_accuracies, label='test acc')
ax2.set_ylim(0, 100)
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy (%)')
ax2.set_title('accuracy with uns. training')
ax2.legend()

# Save the figure as a PDF file
fig.savefig(dir_files + '/linear_classif_occ.pdf')


# %% [markdown]
# - The code appends the average training and testing metrics (accuracy and loss) obtained during the current epoch to their respective lists.
# - It then saves the accumulated metrics (lists) to a file using the **`torch.save`** function. The metrics are saved as a dictionary with keys such as **`'train_accuracies'`**, **`'test_accuracies'`**, **`'train_losses'`**, and **`'test_losses'`**. The saved file path is determined by **`dir_files + '/' + acc_file`**.
# - A figure is created using **`plt.figure(figsize=(10, 5))`** to plot the metrics.
# - The figure is divided into two subplots: one for the loss plot and the other for the accuracy plot.
# - In the loss subplot (**`ax1`**), the training and testing losses are plotted against the number of epochs (**`e`**). Labels and titles are set, and a legend is added.
# - In the accuracy subplot (**`ax2`**), the training and testing accuracies are plotted against the number of epochs. The y-axis limit is set to 0-100 to represent percentages. Similar to the loss subplot, labels, title, and legend are added.
# - Finally, the figure is saved as a PDF file using **`fig.savefig(dir_files + '/linear_classif_occ.pdf')`**.
# 
# The code essentially accumulates the average training and testing metrics over multiple epochs and saves them for analysis. Additionally, it plots the training and testing losses as well as the training and testing accuracies over the epochs, providing a visual representation of the model's performance during unsupervised training.

# %% [markdown]
# 


