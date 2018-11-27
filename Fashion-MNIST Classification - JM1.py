#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 06:46:27 2018

@author: Jake
"""

'''
Classifying the Fashion-MNIST dataset using Pytorch

First full attempt at creating CNN after DL0110EN: 
    Deep Learning with Python and PyTorch
    on edX

'''

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import numpy as np
from collections import OrderedDict

# Utility/helper functions
def show_image(image):
    '''
    Displays 28x28 pixel image with correct classification
    '''
    clothing_class = index_to_class(int(image[1]))
    plt.imshow(image[0].numpy().reshape(28,28))
    plt.title('Correct class: ' + clothing_class +
              ' (' + str(image[1].item()) + ')')

def index_to_class(index):
    '''
    Returns class of item based on index
    '''
    class_dict = {0 : 'T-shirt/top',
                  1 : 'Trouser',
                  2 : 'Pullover',
                  3 : 'Dress',
                  4	: 'Coat',
                  5	: 'Sandal',
                  6	: 'Shirt',
                  7	: 'Sneaker',
                  8	: 'Bag',
                  9 : 'Ankle boot'}
    return(class_dict[index])

def plot_parameters(weights, title):
    '''
    Plots out kernel parameters for visualization
    '''
    n_filters = weights.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_filters)))

    min_value = weights.min().item()
    max_value = weights.max().item()
    
    fig, ax = plt.subplots(n_rows, ncols = n_filters//n_rows)
    fig.subplots_adjust(wspace = 1.0, hspace = 1.0)
    
    for i, ax in enumerate(ax.flat):
        ax.set_xlabel('Kernel: ' + str(i + 1))
        ax.imshow(weights[i][0], vmin = min_value, vmax = max_value,
                  cmap = 'plasma')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title)
    plt.show()

# Load data
train_dataset = dsets.FashionMNIST(root = './JM1',
                                   train = True,
                                   download = True,
                                   transform = transforms.ToTensor())
validation_dataset = dsets.FashionMNIST(root = './JM1',
                                        train = False,
                                        download = True,
                                        transform = transforms.ToTensor())

model = nn.Sequential(OrderedDict([
        ('conv1',       nn.Conv2d(in_channels = 1, out_channels = 16,
                                  kernel_size = 5, stride = 1, padding = 2)),
        ('relu1',       nn.ReLU()),
        ('maxpool1',    nn.MaxPool2d(kernel_size = 2)),
        ('conv2',       nn.Conv2d(in_channels = 16, out_channels = 32,
                                  kernel_size = 5, stride = 1, padding = 2)),
        ('relu2',       nn.ReLU()),
        ('maxpool2',    nn.MaxPool2d(kernel_size = 2)),
        ('dense1',      nn.Linear(32 * 7 * 7, 10))])) # multiplication?

# Testing
rand_num = np.random.randint(0,10)

show_image(train_dataset[rand_num])
plot_parameters(model.state_dict()['conv1.weight'],
                'First Convolutional Weights')
plot_parameters(model.state_dict()['conv2.weight'],
                'Second Convolutional Weights')