# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:58:40 2023

@author: lyork
"""

import numpy as np

def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def compute_feature(self, ii):
    return ii[self.y+self.height][self.x+self.width] + ii[self.y]
    [self.x] - (ii[self.y+self.height][self.x]+ii[self.y]
    [self.x+self.width])

#Viola Jones uses an AdaBoost variant. This is used to help one classifier correct the 
#mistakes of the previous classifier

#Initialize the weights
#Normalize the weights
#Pick the best weak classifier based on the weighted error
#Updated the weights based on the chosen classifiers error
#Repeat until T is the desired number of weak classifiers
class ViolaJones:
    def __init__(self, T=10):
        self.T = T

def train(self, training):
    training_data = []
    for x in range(len(training)):
        training_data.append((integral_image(training[x][0]),
    training[x][1]))

def train(self, training, pos_num, neg_num):
    weights = np.zeros(len(training))
    training_data = []
    for x in range(len(training)):
        training_data.append((integral_image(training[x][0]), training[x][1]))
        if training[x][1] == 1:
            weights[x] = 1.0 / (2 * pos_num)
        else:
            weights[x] = 1.0 / (2 * neg_num)
            

    
    
