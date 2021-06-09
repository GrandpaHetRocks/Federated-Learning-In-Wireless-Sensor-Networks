# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:02:59 2021

@author: Ayush
"""
from Dataset import load_dataset,getImage
import matplotlib.pyplot as plt

trainset,testset,train_group,test_group=load_dataset(10,"noniid")

first_client_batches=getImage(trainset,train_group[0],64)

# =============================================================================
# test nature of distribution
#  print(len(train_group[0]))
#  print(len(train_group[2]))
# =============================================================================


images,labels=next(iter(first_client_batches))
for i in range (64):
    plt.imshow(images[i].numpy().squeeze(), cmap='Greys_r')