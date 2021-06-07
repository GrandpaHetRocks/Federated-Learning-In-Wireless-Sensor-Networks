# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:06:45 2021

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
        self.hidden_new=nn.Linear(512,512)  #adding another layer
        self.output=nn.Linear(512,10) #10 outputs as mnist has 10 classes
        self.relu=nn.ReLU() #using ReLU activation fumction
        self.softmax=nn.LogSoftmax(dim=1) #using softmax activation for multiple outputs
        
    def forward(self,x):
        x=self.hidden(x)
        x=self.relu(x)
        x=self.hidden_new(x)
        x=self.relu(x)
        x=self.output(x)
        x=self.softmax(x)
        return(x)

model=Neural_Net()
print(model)

criterion=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for epoch in range(5): #running 5 epochs
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1) #convert 64 images (in a batch) into single row matrix
        optimizer.zero_grad() #sets gradients to 0
        logits = model(images) #forward propagation calcualting log probabilities
        loss = criterion(logits, labels) #calculates loss
        loss.backward() #backward propagation
        optimizer.step() #updates weights
        running_loss += loss.item() 
    else:
        print('The running loss is: {}'.format(running_loss/len(trainloader)))

images,labels=next(iter(trainloader))
img=images[0].view(1,-1) #just one image
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')

with torch.no_grad():  #we dont want the prediction
    logprobs=model(img) #inference

print("log probabilities:",logprobs) #10 classes in MNIST
print("Actual Probabilities:",torch.exp(logprobs))
print("prediction:",torch.argmax(logprobs))
    


