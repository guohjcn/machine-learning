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


def data_regression():
    """回归模型
    几个关键参数有n_samples（生成样本数）， n_features（样本特征数），noise（样本随机噪音）和coef（是否返回回归系数）
    """
    from sklearn.datasets.samples_generator import make_regression
    X,y, coef = make_regression(n_samples=1000, n_features=1,noise=10, coef=True)
    # X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
    plt.scatter(X,y)
    plt.plot(X,X*coef)
    plt.title("data regression")
    plt.show()

def data_classification():
    """分类模型
    几个关键参数有n_samples（生成样本数）， n_features（样本特征数）， n_redundant（冗余特征数）和n_classes（输出的类别数）
    """    
    from sklearn.datasets.samples_generator import make_classification
    plt.title("data classification")
    

    X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,n_clusters_per_class=1, n_classes=3)
    # X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
    plt.subplot(2,2,1)
    plt.scatter(X1[:,0], X1[:,1])
    plt.xlabel("")
    
    X2, Y2 = make_classification(n_samples=400, n_features=2, n_redundant=0,n_clusters_per_class=1, n_classes=3)
    plt.subplot(2,2,2)


    plt.subplot(2,2,3)
    plt.subplot(2,2,4)
    
    plt.show()

def data_cluster():
    """聚类模型
    几个关键参数有n_samples（生成样本数）， n_features（样本特征数），centers(簇中心的个数或者自定义的簇中心)和cluster_std（簇数据方差，代表簇的聚合程度）
    """
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=1000, n_features=2, 
            centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
    plt.scatter(X[:,0], X[:,1])
    plt.title("data cluster/blobs")
    plt.show()



def main():
    #data_regression()
    data_classification()
    #data_cluster()

if __name__ == '__main__':
    main()
