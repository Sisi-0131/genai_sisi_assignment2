import torch
from torch import nn


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ReLU activation
        self.relu1 = nn.ReLU()

        # First max-pooling layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        
        # Convert feature maps into one-dimensional vectors
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(
            in_features=32 * 16 * 16,
            out_features=100,
        )

        # ReLU activation
        self.relu3 = nn.ReLU()

        # Final output layer
        self.fc2 = nn.Linear(
            in_features=100,
            out_features=10,
        )

    def forward(self, images):
        output = self.conv1(images)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)

        output = self.flatten(output)
        output = self.fc1(output)
        output = self.relu3(output)
        output = self.fc2(output)

        return output
    