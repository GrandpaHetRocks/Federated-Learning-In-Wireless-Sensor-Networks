# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:57:03 2021

@author: Ayush
"""

import torch, torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])


trainset=torchvision.datasets.MNIST(root="./", train= False, transform=transform, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

for images, labels in trainloader:
    print(images.size(), labels.size())
    break

batches = iter(trainloader)
one_batch = next(batches)
images, labels = one_batch
len(images)
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
