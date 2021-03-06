#!usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os
print("Python version:", sys.version)

import time
from sklearn import metrics
import numpy as np
import pickle
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification

def generate_3d_data():
    X, y = make_classification(n_samples=100, n_features=3, n_redundant=0, n_classes=4, n_informative=2,
                                n_clusters_per_class=1, class_sep=0.5,random_state=10)
    
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)
    ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', c=y)
    #X, y = make_classification()
    #plt.scatter(X[:,0], X[:,1], X[:,2], marker='o', c=y)
    plt.show()

generate_3d_data()