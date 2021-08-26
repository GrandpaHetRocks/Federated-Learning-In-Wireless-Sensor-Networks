# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 00:54:49 2021

@author: Ayush
"""

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot
import pandas as pd
import math
import numpy as np
import random
from sklearn.datasets import make_blobs


def calc_distance(X1, X2):
    return ((sum((X1 - X2)**2))**0.5)  #increase spread here


def path_loss_calc(clients):
    path_loss_list=[]
    dis_list=[]
    for i in range(len(clients)-1):
        for j in range(i+1,len(clients)):
            X1=clients['client'+str(i+1)]
            X2=clients['client'+str(j+1)]
            dis=calc_distance(X1,X2)
            #path_loss=60*math.exp(-dis)+random.uniform(-5,5)
            path_loss=10*math.log10(10000/(dis*dis))
            #print(path_loss)
            path_loss_list.append(['client'+str(i+1),'client'+str(j+1),path_loss])
            dis_list.append(dis)
    # pl=[]
    # for ki in path_loss_list:
    #     pl.append(ki[2])
    # pl.sort()
    # print(pl)
    return(path_loss_list,dis_list)

def noise(clients):
    noise_list=[]
    for i in range(len(clients)-1):
        for j in range(i+1,len(clients)):
            noise=random.uniform(0,5)
            noise_list.append(['client'+str(i+1),'client'+str(j+1),noise])
            
    return(noise_list)


def get_cluster(number=15):
    cluster_array, _ = make_classification(n_samples=number, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=50)
    #cluster_array, _ = make_blobs(n_samples=number,n_features=2, centers=2)
    no=1
    clients={}
    for client in cluster_array:
        clients['client'+str(no)]=client
        no+=1
    path_loss,dis=path_loss_calc(clients)
    noise_list=noise(clients)
    
    for i in cluster_array:
        pyplot.scatter(i[0],i[1],c='red')
    
    ch=random.randint(1,number)
    cluster_head=clients['client'+str(ch)]
    pyplot.scatter(cluster_head[0],cluster_head[1],c='green')
    
    snr_list=[]
    for i in range (len(path_loss)):
        if('client'+str(ch) in path_loss[i]):
            snr_list.append(path_loss[i][2]-noise_list[i][2])
    
    snr_list.sort()
    print(snr_list)
    #print(cluster_head)
    pyplot.show()
    return(snr_list,'client'+str(ch))   

#get_clusters()
