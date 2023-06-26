# %%
import torch  # Import the torch library
import torch.nn as nn  # Import the torch.nn module
import torchvision.models as models  # Import the models module from torchvision
import torch.nn.functional as F  # Import the torch.nn.functional module
from torch.nn.functional import adaptive_avg_pool2d  # Import the adaptive_avg_pool2d function from torch.nn.functional

# %%
class Generator(nn.Module):
    def __init__(self, ngpu, latent_dim, ngf=64, img_channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent_dim = latent_dim
        self.bias = True

        # Define the transposed convolution layers
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*4, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size: (ngf*4) x 4 x 4

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size: (ngf*2) x 8 x 8

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size: (ngf) x 16 x 16

        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.Tanh()
        )
        # state size: (img_channels) x 32 x 32

    def forward(self, input, reverse=True):
        fc1 = input.view(input.size(0), input.size(1), 1, 1)
        # Reshape the input tensor

        tconv1_output = self.tconv1(fc1)
        # Apply the first transposed convolution layer

        tconv2_output = self.tconv2(tconv1_output)
        # Apply the second transposed convolution layer

        tconv3_output = self.tconv3(tconv2_output)
        # Apply the third transposed convolution layer

        output = self.tconv4(tconv3_output)
        # Apply the fourth transposed convolution layer

        if reverse:
            output = grad_reverse(output)
            # Reverse the gradient of the output (custom function)

        return output

# %% [markdown]
# 1. The `Generator` class is defined as a subclass of `nn.Module`, which is the base class for all neural network modules in PyTorch.
# 
# 2. In the `__init__` method, the constructor initializes the generator by specifying the number of GPUs (`ngpu`), the dimension of the input noise vector (`latent_dim`), the number of filters in the generator's convolutional layers (`ngf`), and the number of channels in the output image (`img_channels`).
# 
# 3. The generator uses transposed convolution layers (`nn.ConvTranspose2d`) to upsample the input noise and generate images. The transposed convolution layers are defined and initialized in the constructor.
#     - `self.tconv1` is the first transposed convolution layer that takes the input noise and produces feature maps with `ngf*4` channels. It uses a kernel size of 4, stride of 1, and no padding. It is followed by a leaky ReLU activation function (`nn.LeakyReLU`).
#     - `self.tconv2` is the second transposed convolution layer that takes the output of `self.tconv1` and produces feature maps with `ngf*2` channels. It uses a kernel size of 4, stride of 2, and padding of 1. It is also followed by a leaky ReLU activation function.
#     - `self.tconv3` is the third transposed convolution layer that takes the output of `self.tconv2` and produces feature maps with `ngf` channels. It uses the same configuration as `self.tconv2`.
#     - `self.tconv4` is the final transposed convolution layer that takes the output of `self.tconv3` and produces the final output image. It uses a kernel size of 4, stride of 2, and padding of 1. It is followed by a hyperbolic tangent (`nn.Tanh`) activation function to ensure the output values are within the range [-1, 1].
#     
# 4. The `forward` method implements the forward pass of the generator. It takes an input tensor (`input`) and a boolean flag (`reverse`) to indicate whether to reverse the gradient of the output.
#     - The input tensor is reshaped (`view`) to have dimensions (batch_size, latent_dim, 1, 1), where `latent_dim` is the dimension of the input noise vector.
#     - The reshaped tensor is passed through each transposed convolution layer (`self.tconv1`, `self.tconv2`, `self.tconv3`, `self.tconv4`) sequentially.
#     - If the `reverse` flag is `True`, the gradient of the output tensor is reversed using a custom function called `grad_reverse` (not defined in the provided code). This is often used in domain adaptation tasks to fool the discriminator by making the generator's output look more like the target domain.
#     - The final output tensor is returned.
# 
# Overall, this code defines the architecture of a generator model for a GAN, which generates synthetic images from random noise.

# %%
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        # The forward pass returns the input tensor as is
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        # The backward pass multiplies the gradient output by -1
        return (grad_output * -1)

def grad_reverse(x):
    # Apply the gradient reversal operation using the custom GradReverse function
    return GradReverse.apply(x)

# %% [markdown]
# 1. The code defines a custom autograd function called `GradientReverse`. This function will be used to reverse the gradient during the backpropagation process.
# 
# 2. The `forward` method of  `GradientReverse ` takes an input tensor  `x ` and returns it as is, without any modifications. This is the identity operation.
# 
# 3. The  `backward ` method of  `GradientReverse ` takes the gradient of the output with respect to the forward pass and multiplies it by -1. This effectively reverses the gradient direction.
# 
# 4. The  `grad_reverse ` function is defined to apply the gradient reversal operation. It calls the  `apply ` method of  `GradientReverse ` to perform the operation on the input tensor  `x `.
# 
# By using the  `grad_reverse ` function, you can reverse the gradients during the backpropagation process, which can be useful for tasks such as domain adaptation in adversarial learning setups.

# %%
class Flatten(torch.nn.Module):
    def forward(self, input):
        # Retrieve the batch size from the input tensor
        batch_size = input.shape[0]

        # Reshape the input tensor to have shape (batch_size, -1)
        flattened_input = input.view(batch_size, -1)

        # Return the flattened input tensor
        return flattened_input

# %% [markdown]
# 1. The code defines a custom module called  `Flatten `, which inherits from  `torch.nn.Module `. This module is responsible for flattening the input tensor.
# 
# 2. The  `forward ` method of  `Flatten ` is overridden to define the forward pass operation.
# 
# 3. Inside the  `forward ` method, the batch size is obtained from the input tensor using  `x.shape[0] `. This represents the number of samples in the batch.
# 
# 4. The input tensor is then reshaped using  `x.view(batch_size, -1) `. The  `view ` function is used to reshape the tensor, where  `batch_size ` represents the number of samples in the batch, and  `1 ` infers the correct size for the remaining dimensions.
# 
# 5. The flattened input tensor is returned as the output of the forward pass.
# 
# The purpose of the  `Flatten ` module is to flatten multi-dimensional input tensors into a 2D representation, which is commonly required when transitioning from convolutional layers to fully connected layers in neural networks.

# %%
class Discriminator(nn.Module):
    def __init__(self, ngpu, latent_dim, ndf=64, img_channels=3, dropout_prob=0.0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.bias = True

        # Input size: (img_channels) x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output size: (ndf) x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output size: (ndf*2) x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output size: (latent_dim) x 4 x 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, latent_dim, kernel_size=4, stride=2, padding=0, bias=self.bias),
            Flatten()
        )

        # Output size: 1 x 1
        self.dis = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0, bias=self.bias),
            Flatten()
        )

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Pass input through the convolutional layers
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform the real/fake classification
        fc_dis = self.sigmoid(self.dis(conv3))

        # Extract the encoded feature representation
        fc_enc = self.conv4(conv3)

        # Reshape the real/fake classification tensor
        realfake = fc_dis.view(-1, 1).squeeze(1)

        return fc_enc, realfake


# %% [markdown]
# 1. The code defines a discriminator model for a GAN. The discriminator takes an input image and outputs both an encoded feature representation and a real/fake classification.
# 
# 2. The  `Discriminator ` class inherits from  `nn.Module ` and defines the discriminator architecture.
# 
# 3. The constructor ( `__init__ ` method) initializes the discriminator by specifying the number of GPUs ( `ngpu `), the dimension of the encoded feature representation ( `latent_dim `), the number of filters in the discriminator's convolutional layers ( `ndf `), the number of channels in the input image ( `img_channels `), and the dropout probability ( `dropout_prob `).
# 
# 4. The discriminator uses convolutional layers ( `nn.Conv2d `) followed by leaky ReLU activation functions ( `nn.LeakyReLU `) to process the input image and extract features.
# 
# 5. The discriminator has four convolutional layers ( `self.conv1 `,  `self.conv2 `,  `self.conv3 `,  `self.conv4 `) that progressively downsample the input image.
# 
# 6. The  `self.dis ` layer performs a convolution followed by flattening to output the real/fake classification.
# 
# 7. The  `self.conv4 ` layer performs a convolution followed by flattening to output the encoded feature representation.

# %%
class OutputClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_classes=10):
        super(OutputClassifier, self).__init__()

        # Define the fully connected layer for classification
        self.fc_classifier = nn.Sequential(
            nn.Linear(input_size, num_classes, bias=True),
        )

        # Softmax activation function for class probabilities
        self.softmax = nn.Softmax()

    def forward(self, input):
        # Perform classification using the fully connected layer
        classes = self.fc_classifier(input)

        return classes

# %% [markdown]
# 1. The code defines an output classifier model that takes an input and predicts the class labels.
# 
# 2. The  `OutputClassifier ` class inherits from  `nn.Module ` and defines the classifier architecture.
# 
# 3. The constructor ( `__init__ ` method) initializes the classifier by specifying the input size ( `input_size `), the size of the hidden layer ( `hidden_size `), and the number of classes ( `num_classes `).
# 
# 4. The classifier consists of a single fully connected layer ( `nn.Linear `) that maps the input to the number of classes. This layer performs the classification task.
# 
# 5. The  `self.softmax ` layer applies the softmax activation function to the output of the classifier. This converts the output logits into class probabilities.
# 
# 6. The forward pass is implemented in the  `forward ` method. It takes an input, performs classification using the fully connected layer, and returns the predicted classes.
# 
# Please note that the  `softmax ` function is typically not applied within the model, as it is usually incorporated in the loss function or evaluation metric outside the model.

# %%
class InputClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(InputClassifier, self).__init__()

        # Define the fully connected layer for classification
        self.fc_classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes, bias=True),
        )

    def forward(self, input):
        # Reshape the input from batch_size x 28 x 28 to batch_size x (28*28)
        out = input.view(input.size(0), -1)

        # Perform classification using the fully connected layer
        out = self.fc_classifier(out)

        return out

# %% [markdown]
# 1. The code defines an input classifier model that takes an input and predicts the class labels.
# 
# 2. The  `InputClassifier ` class inherits from  `nn.Module ` and defines the classifier architecture.
# 
# 3. The constructor ( `__init__ ` method) initializes the classifier by specifying the input dimension ( `input_dim `) and the number of classes ( `num_classes `).
# 
# 4. The classifier consists of a single fully connected layer ( `nn.Linear `) that maps the input to the number of classes. This layer performs the classification task.
# 
# 5. The forward pass is implemented in the  `forward ` method. It takes an input and reshapes it from batch_size x 28 x 28 to batch_size x (28*28). This flattens the input image into a 1D vector.
# 
# 6. The flattened input is then passed through the fully connected layer ( `self.fc_classifier `), which applies the linear transformation (input * A + b) where A and b are learnable parameters of the linear layer.
# 
# 7. The output  `out ` represents the logits or scores for each class. The final predicted class can be obtained by applying a softmax activation function or using an appropriate loss function during training.
# 
# Please note that the code assumes the input shape to be batch_size x 28 x 28, which is a common representation for images in MNIST-like datasets.

# %%
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    DEFAULT_BLOCK_INDEX = 3  # Index of default block of inception to return

    # Maps feature dimensionality to their output block indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, 'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        # Load the pretrained InceptionV3 model
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        # Set the requires_grad property of parameters
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, input):
        """Get Inception feature maps"""
        output = []
        x = input

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)

            if idx == self.last_needed_block:
                break

        return


# %% [markdown]
# The code defines a PyTorch module called `InceptionV3`, which represents a pretrained InceptionV3 network returning feature maps.
# 
# The key components and functionalities of the code are as follows:
# 
# 1. The class variable `DEFAULT_BLOCK_INDEX` is set to 3, representing the index of the default block of the Inception network to return.
# 
# 2. The dictionary `BLOCK_INDEX_BY_DIM` maps feature dimensionality to their respective output block indices.
# 
# 3. The `__init__` method initializes the `InceptionV3` module. It takes several parameters:
#    - `output_blocks`: A list of output block indices to return. The default is the `DEFAULT_BLOCK_INDEX` (3).
#    - `resize_input`: A boolean indicating whether to resize the input. The default is `True`.
#    - `normalize_input`: A boolean indicating whether to normalize the input. The default is `True`.
#    - `requires_grad`: A boolean indicating whether the model's parameters require gradient computation. The default is `False`.
# 
# 4. The method loads the pretrained InceptionV3 model using `models.inception_v3(pretrained=True)`.
# 
# 5. The different blocks of the InceptionV3 model are constructed and stored in `self.blocks` using `nn.Sequential`.
#    - Block 0 corresponds to the input to maxpool1 and consists of several convolutional layers and a max pooling layer.
#    - Block 1 corresponds to maxpool1 to maxpool2 and consists of convolutional layers and a max pooling layer.
#    - Block 2 corresponds to maxpool2 to the auxiliary classifier and consists of several mixed layers.
#    - Block 3 corresponds to the auxiliary classifier to the final average pooling layer.
#    
# 6. The `forward` method performs the forward pass of the InceptionV3 network. It takes an input tensor `input` and returns a list of feature maps corresponding to the selected output blocks.
#    - The input tensor is resized and normalized if specified.
#    - The input tensor is passed through the blocks sequentially, and the output feature maps are stored in the `output` list.
#    - The method breaks the loop when it reaches the last needed block.
# 
# Overall, this code allows you to use the pretrained InceptionV3 model to extract specific feature maps based on the desired output blocks.


