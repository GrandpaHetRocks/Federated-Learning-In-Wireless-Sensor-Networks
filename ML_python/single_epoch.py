# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 01:32:44 2021

@author: Ayush
"""

import torch, torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor()])


trainset=torchvision.datasets.MNIST(root="./", train= False, transform=transform, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(28*28,512) #28*28 is image size, 512 neurons (convention in powers of 2)
        self.output=nn.Linear(512,10) #10 outputs as mnist has 10 classes
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x=self.hidden(x)
        x=self.sigmoid(x)
        x=self.output(x)
        x=self.softmax(x)
        return(x)

model=Neural_Net()
print(model)

criterion=nn.NLLLoss()
images,labels=next(iter(trainloader))
images=images.view(images.shape[0],-1)
logits=model(images) #forward propagation calcualting log probabilities
loss=criterion(logits,labels)
print(loss)
loss.backward() #go backward and update weights
optimizer=optim.SGD(model.parameters(),lr=0.01)
optimizer.step()
