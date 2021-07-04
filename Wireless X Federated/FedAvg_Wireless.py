# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:32:14 2021

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
import random
import math
from bitstring import *

P=2 #signal power
#stream = BitStream()
key=[]
for i in range (10000):
    temp=random.randint(0,1)
    key.append(temp)

key1=[0]*len(key)
for i in range (len(key)):   #bpsk modulation
    if(key[i]==1):
        #print("yay")
        key1[i]=-math.sqrt(P)
    else:
        key1[i]=math.sqrt(P)

#print(key)
        
key_np=np.array(key1)

#print(key)
#key=np.fromstring(key)
#key1=key.*5
# for i in range (10000):
#     temp=random.randint(0,1)
#     key1.append(temp)
#print(np.bitwise_xor(key, key1))
class Arguments():
    def __init__(self):
        self.images = 10000
        self.clients = 30
        self.rounds = 2
        self.epochs = 5
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

for inx, client in enumerate(clients):  #return actual image set for each client
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
        #self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #x=self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def ClientUpdate(args, device, client,key_np,key):
    gc=False
    client['model'].train()
    #simulating a wireless channel
    snr=random.randint(-10,40)   #tamper here to make the channel good/bad :P
    #snr=40
    print("SNR= ",snr)
    snr__=10**(snr/10)
    std=math.sqrt(P/snr__) #channel noise
    x=random.random()
    y=random.random()
    h=complex(x,y)
    
    
    data=client['model'].conv1.weight
    data=data*math.sqrt(P) #transmitted signal
    data=h*data+(torch.randn(data.size())*std) #channel affecting data
    data=data/(math.sqrt(P)*(h))  #demodulating received data
    data=data.real #demodulating received data
    client['model'].conv1.weight.data=data
    
    data=client['model'].conv2.weight
    data=data*math.sqrt(P) #transmitted signal
    data=h*data+(torch.randn(data.size())*std) #channel affecting data
    data=data/(math.sqrt(P)*(h))  #demodulating received data
    data=data.real #demodulating received data
    client['model'].conv2.weight.data=data
    
    #print(client['model'].conv1.weight.size)
    client['model'].send(client['hook'])
    print("Client:",client['hook'].id)
    
    key_np_received=h*key_np+(np.random.randn(len(key_np))*std*2)
    #print(key_np_received)
    key_np_received=(key_np_received/(h)).real
    
    for o in range (len(key_np_received)):  #demodulation bpsk
        if(key_np_received[o]>=0):
            key_np_received[o]=0
        else:
            key_np_received[o]=1
    
    key_np_received=key_np_received.tolist()
    key_np_received = [int(item) for item in key_np_received]
    #key_np=key_np.tolist()
    
    #print(key_np_received)
    #print(key==key_np_received)
    #print(sum(np.bitwise_xor(key,key_np_received))/len(key))
    
    
    
    
    
    if(sum(np.bitwise_xor(key,key_np_received))/len(key)==0): #...............................................checking if channel is good enough for transmission by checking BER..................................
        gc=True #considering the client model for averaging
        for epoch in range(1, args.epochs + 1):
            
            # snr=random.randint(-20,40)
            # #snr=40
            # print("SNR= ",snr)
            # snr__=10**(snr/10)
            # std=math.sqrt(P/snr__) #channel noise
            # x=random.random()
            # y=random.random()
            # h=complex(x,y)
            #h=math.sqrt(P)*(h)
            #h=math.sqrt(P)*abs(h)
            for batch_idx, (data, target) in enumerate(client['trainset']): 
                # print(data)
                # data1=data.numpy()
                # print(data1)
                # print(data1)
                # for i in range (len(data1)):
                #     for j in range(len(data1[i])):
                #         for k in range(len(data1[i][j])):
                #             b = Bits().join(Bits(float=x, length=64) for x in data1[i][j][k])
                
                
                # data=data*math.sqrt(P) #transmitted signal
                # data=h*data+(torch.randn(data.size())*std) #channel affecting data
                # data=data/(math.sqrt(P)*(h))  #demodulating received data
                # data=data.real #demodulating received data
                
                
                # if(snr>-100): 
                    #print("SNR=",snr)
                data = data.send(client['hook'])
                target = target.send(client['hook'])
                
                #train model on client
                data, target = data.to(device), target.to(device) #send data to cpu/gpu (data is stored locally)
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
    else:
        print("Poor Channel, client not taken for averaging in this round")
            
                    
    client['model'].get()
    print()
    return gc
    

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
    #client['model'] = torch.quantization.quantize_dynamic(
    # client['model'],  # the original model
    # {torch.nn.Linear},  # a set of layers to dynamically quantize
    # dtype=torch.fp)  # the target dtype for quantized weights
    client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)
    


for fed_round in range(args.rounds):
    
    client_good_channel=[] #to check which clients have a good channel, only those will be taken for averaging per round
    
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
    #even slot
    for client in active_clients:
        goodchannel=ClientUpdate(args, device, client,key_np,key)
        if(goodchannel):
            client_good_channel.append(client)
    
#     # Testing 
#     for client in active_clients:
#         test(args, client['model'], device, client['testset'], client['hook'].id)
    
    
    # Averaging 
        #odd slot
    print()
    print("Clients having a good channel and considered for averaging")
    for no in range (len(client_good_channel)):
        print(client_good_channel[no]['hook'].id)
    global_model = averageModels(global_model, client_good_channel)
    #global_model = averageModels(global_model, active_clients)
    #print(active_clients==client_good_channel)
    
    # Testing the average model
    test(args, global_model, device, global_test_loader, 'Global')
            
    # Share the global model with the clients
    for client in clients:
        client['model'].load_state_dict(global_model.state_dict())
        #client['model']=torch.quantization.quantize_dynamic(client['model'],{torch.nn.Conv2d},dtype=torch.qint8)
        #print(client['model'].conv1.weight.data)
        
if (args.save_model):
    torch.save(global_model.state_dict(), "FedAvg.pt")
    
