import torch
import torch.nn as nn
from typing import Optional, Tuple, Iterable


class LinearParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, hidden_sizes=(10, 5, 5, 10), dropout=0.5, **kwargs):
        """Parameterizer for compas dataset.
        
        Solely consists of fully connected modules.

        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        hidden_sizes : iterable of int
            Indicates the size of each layer in the network. The first element corresponds to
            the number of input features.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(h, h_next))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of compas parameterizer.

        Computes relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        return self.layers(x).view(x.size(0), self.num_concepts, self.num_classes)



class ConvParameterizer(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        cl_sizes: Iterable[int] = (3, 10, 20),
        kernel_size: int = 5,
        hidden_sizes: Iterable[int] = (169, 256, 64),  # default to some reasonable size
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        input_image_size=(256, 192),   # your actual input size here!
        **kwargs
    ):
        """
        Parameterizer for concept relevances, typically for image data like MNIST.

        Parameters
        ----------
        num_concepts : int
            Number of concepts to parameterize.
        num_classes : int
            Number of output classes.
        cl_sizes : iterable of int
            Number of channels for each conv layer (first is input channels).
        kernel_size : int
            Kernel size for conv layers.
        hidden_sizes : iterable of int
            Sizes of fully connected layers. First must match flattened conv output size.
        dropout : float
            Dropout probability.
        use_batchnorm : bool
            Whether to add BatchNorm after conv layers.
        input_image_size : tuple (height, width), optional
            Input image spatial size to calculate fc layer input size automatically.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        # Build convolutional layers
        cl_layers = []
        for in_ch, out_ch in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size)
            )
            if use_batchnorm:
                cl_layers.append(nn.BatchNorm2d(out_ch))
            cl_layers.append(nn.Dropout2d(dropout))
            cl_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            cl_layers.append(nn.ReLU(inplace=True))
        self.cl_layers = nn.Sequential(*cl_layers)

        # If input_image_size provided, compute flattened size automatically
        if input_image_size is not None:
            with torch.no_grad():
                dummy = torch.zeros(1, cl_sizes[0], *input_image_size)
                conv_out = self.cl_layers(dummy)
                flattened_size = conv_out.view(1, -1).shape[1]
        else:
            # Require user to specify correct hidden_sizes[0] for input
            flattened_size = hidden_sizes[0]

        # Build fully connected layers
        fc_layers = []
        prev_size = flattened_size
        for size in hidden_sizes[1:]:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.ReLU(inplace=True))
            prev_size = size

        # Final output layer maps to num_concepts * num_classes
        fc_layers.append(nn.Linear(prev_size, num_concepts * num_classes))
        fc_layers.append(nn.Tanh())  # Output activation

        self.fc_layers = nn.Sequential(*fc_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores of shape (batch_size, num_concepts, num_classes).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W) but got shape {x.shape}")

        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        fc_output = self.fc_layers(flattened)
        return fc_output.view(-1, self.num_concepts, self.num_classes)
