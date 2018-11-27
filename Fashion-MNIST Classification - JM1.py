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
import numpy

# Utility/helper functions
def show_image(image):
    '''
    Displays 28x28 pixel image with correct classification
    '''
    clothing_class = index_to_class(int(image[1]))
    plt.imshow(image[0].numpy().reshape(28,28))
    plt.title('Correct class: ' + clothing_class)

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

# Load data
train_dataset = dsets.FashionMNIST(root = './JM1',
                                   train = True,
                                   download = True,
                                   transform = transforms.ToTensor())
validation_dataset = dsets.FashionMNIST(root = './JM1',
                                        train = False,
                                        download = True,
                                        transform = transforms.ToTensor())

show_image(train_dataset[13])