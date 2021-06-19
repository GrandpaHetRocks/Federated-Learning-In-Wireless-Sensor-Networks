# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:32:37 2021

@author: Ayush
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import syft as sy
import copy
import numpy as np
import time
import Dataset
from Dataset import load_dataset, getImage
from utils import averageModels

class Arguments():
    def __init__(self):
        self.images = 10000
        self.clients = 10
        self.rounds = 5
        self.epochs = 2
        self.local_batches = 64
        self.lr = 0.01
        self.C = 0.9 #fraction of clients used in the round
        self.drop_rate = 0.1 #fraction of devices in the selected set to be dropped for various reasons
        self.torch_seed = 0 #same weights and parameters whenever the program is run
        self.log_interval = 64
        self.iid = 'iid'
        self.split_size = int(self.images / self.clients)
        self.samples = self.split_size / self.images 
        self.use_cuda = False
        self.save_model = True

args = Arguments()

#checking if gpu is available
use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

hook = sy.TorchHook(torch)
clients = []

#generating virtual clients
for i in range(args.clients):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i+1))})
    
global_train, global_test, train_group, test_group = load_dataset(args.clients, args.iid) #load data

for inx, client in enumerate(clients):  #return actual image for each client
    trainset_ind_list = list(train_group[inx]) 
    client['trainset'] = getImage(global_train, trainset_ind_list, args.local_batches)
    client['testset'] = getImage(global_test, list(test_group[inx]), args.local_batches)
    client['samples'] = len(trainset_ind_list) / args.images #useful while taking weighted average

#load dataset for global model (to compare accuracies)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
global_test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
global_test_loader = DataLoader(global_test_dataset, batch_size=args.local_batches, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def ClientUpdate(args, device, client):
    client['model'].train()
    client['model'].send(client['hook'])
    
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(client['trainset']): #send image and label to client
            data = data.send(client['hook'])
            target = target.send(client['hook'])
            
            #train model on client
            data, target = data.to(device), target.to(device) #send data to cpu/gpu
            output = client['model'](data)
            loss = F.nll_loss(output, target)
            loss.backward()
            client['optim'].step()
            
            if batch_idx % args.log_interval == 0:
                loss = loss.get() 
                print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    client['hook'].id,
                    epoch, batch_idx * args.local_batches, len(client['trainset']) * args.local_batches, 
                    100. * batch_idx / len(client['trainset']), loss))
                
    client['model'].get()
    

def test(args, model, device, test_loader, name):
    model.eval()    #no need to train the model while testing
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss for {} model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

   
torch.manual_seed(args.torch_seed)
global_model = Net() #redundant code as we don't use it for training: assigns a CNN to the global model

for client in clients: #give the model and optimizer to every client
    torch.manual_seed(args.torch_seed)
    client['model'] = Net().to(device)
    client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)

for fed_round in range(args.rounds):
    
#     uncomment if you want a random fraction for C every round
#     args.C = float(format(np.random.random(), '.1f'))
    
    # number of selected clients
    m = int(max(args.C * args.clients, 1)) #at least 1 client is selected for training

    # Selected devices
    np.random.seed(fed_round)
    selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)#dont choose same client more than once
    selected_clients = [clients[i] for i in selected_clients_inds]
    
    # Active devices
    np.random.seed(fed_round)
    active_clients_inds = np.random.choice(selected_clients_inds, int((1-args.drop_rate) * m), replace=False) #drop clients
    active_clients = [clients[i] for i in active_clients_inds]
    
    # Training 
    for client in active_clients:
        ClientUpdate(args, device, client)
    
#     # Testing 
#     for client in active_clients:
#         test(args, client['model'], device, client['testset'], client['hook'].id)
    
    # Averaging 
    global_model = averageModels(global_model, active_clients)
    
    # Testing the average model
    test(args, global_model, device, global_test_loader, 'Global')
            
    # Share the global model with the clients
    for client in clients:
        client['model'].load_state_dict(global_model.state_dict())
        
if (args.save_model):
    torch.save(global_model.state_dict(), "FedAvg.pt")