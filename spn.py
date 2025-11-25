#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path



class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        # LAYER 0
        self.conv00 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv00)
        self.bn00 = nn.BatchNorm2d(num_features=64)

        self.conv01 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv01)
        self.bn01 = nn.BatchNorm2d(num_features=64)

        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # LAYER 1
        self.conv10 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv10)
        self.bn10 = nn.BatchNorm2d(num_features=128)

        self.conv11 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv11)
        self.bn11 = nn.BatchNorm2d(num_features=128)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # LAYER 2
        self.conv20 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv20)
        self.bn20 = nn.BatchNorm2d(num_features=256)

        self.conv21 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv21)
        self.bn21 = nn.BatchNorm2d(num_features=256)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # LAYER 3
        self.conv30 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv30)
        self.bn30 = nn.BatchNorm2d(num_features=512)

        self.conv31 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv31)
        self.bn31 = nn.BatchNorm2d(num_features=512)

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Final average pooling to get fixed size output of 1x1x512
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # First fully connected layer with ReLU
        self.fc1 = nn.Linear(in_features=512 * 2, out_features=512)
        self.initialise_layer(self.fc1)

        # Final fully connected layer to get 3 output classes with softmax and dropout
        self.fc2 = nn.Linear(in_features=512, out_features=self.class_count)
        self.initialise_layer(self.fc2)


    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> torch.Tensor:
        x0 = F.relu(self.bn00(self.conv00(image0)))
        x0 = F.relu(self.bn01(self.conv01(x0)))
        x0 = self.pool0(x0)
        x0 = F.relu(self.bn10(self.conv10(x0)))
        x0 = F.relu(self.bn11(self.conv11(x0)))
        x0 = self.pool1(x0)
        x0 = F.relu(self.bn20(self.conv20(x0)))
        x0 = F.relu(self.bn21(self.conv21(x0)))
        x0 = self.pool2(x0)
        x0 = F.relu(self.bn30(self.conv30(x0)))
        x0 = F.relu(self.bn31(self.conv31(x0)))
        x0 = self.pool3(x0)
        x0 = self.avg_pool(x0)

        x1 = F.relu(self.bn00(self.conv00(image1)))
        x1 = F.relu(self.bn01(self.conv01(x1)))
        x1 = self.pool0(x1)
        x1 = F.relu(self.bn10(self.conv10(x1)))
        x1 = F.relu(self.bn11(self.conv11(x1)))
        x1 = self.pool1(x1)
        x1 = F.relu(self.bn20(self.conv20(x1)))
        x1 = F.relu(self.bn21(self.conv21(x1)))
        x1 = self.pool2(x1)
        x1 = F.relu(self.bn30(self.conv30(x1)))
        x1 = F.relu(self.bn31(self.conv31(x1)))
        x1 = self.pool3(x1)
        x1 = self.avg_pool(x1)

        x = torch.cat((x0, x1), dim=1)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

