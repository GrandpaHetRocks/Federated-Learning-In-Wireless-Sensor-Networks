# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:02:59 2021

@author: Ayush
"""
from Dataset import load_dataset,getImage
import matplotlib.pyplot as plt
import torch
import syft


trainset,testset,train_group,test_group=load_dataset(10,"iid")

first_client_batches=getImage(trainset,train_group[0],64)

# =============================================================================
# test nature of distribution
#  print(len(train_group[0]))
#  print(len(train_group[2]))
# =============================================================================


images,labels=next(iter(first_client_batches))
for i in range (64):
    plt.imshow(images[i].numpy().squeeze(), cmap='Greys_r')
    
    
hook=syft.TorchHook(torch)


Ri=syft.VirtualWorker(hook,id='Ri')
print(Ri)
x = torch.Tensor([1,2,3])
x = x.send(Ri)
print(x)
print(Ri._objects)
print(x.location == Ri)
print(x.location.id == Ri.id)

x = x.get()
print(x)