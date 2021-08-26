#-*-coding:utf-8-*-
"""
Created on Tue Jun  8 22:28:22 2021

@author:Ayush
"""


import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,Dataset
import numpy as np

def mnistIID(dataset,num_users):#this function randomly chooses 60k/10 (assuming 10 users) images and distributes them in iid fashion among the users.
    num_images=int(len(dataset)/num_users)
    # print(num_images)
    users_dict,indices={},list(range(len(dataset))) #length of dataset is 60k
    for i in range(num_users):
        np.random.seed(i) #starts with the same random number to maiantain similarity across runs
        #np.random.choice selects num_images number of random numbers from 0 to indices
        users_dict[i]=set(np.random.choice(indices,num_images,replace=False)) #set drops repeated items
        indices=list(set(indices)-users_dict[i])
    return users_dict

def mnistnonIID(dataset,num_users,test):#function divides dataset into classes and each client gets random 2 classes to train on
    # classes,images=20,500
    if test:
        classes=100
        images=int(len(dataset)/classes)
        #print(len(dataset))
    classes_indx=[i for i in range(classes)]
    users_dict={i:np.array([]) for i in range(num_users)}
    indices=np.arange(classes*images)
    unsorted_labels=dataset.train_labels.numpy()

    indices_unsortedlabels=np.vstack((indices,unsorted_labels)) #make a container for indices and labels so they move together
    indices_labels=indices_unsortedlabels[:,indices_unsortedlabels[1,:].argsort()]
    indices=indices_labels[0,:]

    for i in range(num_users):
        np.random.seed(i)
        #print(classes_indx)
        temp=set(np.random.choice(classes_indx,3,replace=False)) #random 3 classes to each client
        classes_indx=list(set(classes_indx)-temp)
        for t in temp:
            users_dict[i]=np.concatenate((users_dict[i],indices[t*images:(t+1)*images]),axis=0)
    return users_dict

def mnistnonIIDUnequal(dataset,num_users,test):#calsses are there but each client gets different number of classes to train on
    classes,images=1200,50
    if test:
        classes,images=200,50
    classes_indx=[i for i in range(classes)]
    users_dict={i:np.array([]) for i in range(num_users)}
    indices=np.arange(classes*images)
    unsorted_labels=dataset.train_labels.numpy()

    indices_unsortedlabels=np.vstack((indices,unsorted_labels))
    indices_labels=indices_unsortedlabels[:,indices_unsortedlabels[1,:].argsort()]
    indices=indices_labels[0,:]

    min_cls_per_client=1 #a client has to chose at least one class
    max_cls_per_client=30 #client can't choose more than 30 classes

    random_selected_classes=np.random.randint(min_cls_per_client,max_cls_per_client+1,size=num_users)
    random_selected_classes=np.around(random_selected_classes/sum(random_selected_classes)*classes)
    random_selected_classes=random_selected_classes.astype(int)

    if sum(random_selected_classes)>classes:
        for i in range(num_users):
            np.random.seed(i)
            temp=set(np.random.choice(classes_indx,1,replace=False))
            classes_indx=list(set(classes_indx)-temp)
            for t in temp:
                users_dict[i]=np.concatenate((users_dict[i],indices[t*images:(t+1)*images]),axis=0)

        random_selected_classes=random_selected_classes-1

        for i in range(num_users):
            if len(classes_indx)==0:
                continue
            class_size=random_selected_classes[i]
            if class_size>len(classes_indx):
                class_size=len(classes_indx)
            np.random.seed(i)
            temp=set(np.random.choice(classes_indx,class_size,replace=False))
            classes_indx=list(set(classes_indx)-temp)
            for t in temp:
                users_dict[i]=np.concatenate((users_dict[i],indices[t*images:(t+1)*images]),axis=0)
    else:

        for i in range(num_users):
            class_size=random_selected_classes[i]
            np.random.seed(i)
            temp=set(np.random.choice(classes_indx,class_size,replace=False))
            classes_indx=list(set(classes_indx)-temp)
            for t in temp:
                users_dict[i]=np.concatenate((users_dict[i],indices[t*images:(t+1)*images]),axis=0)

        if len(classes_indx)>0:
            class_size=len(classes_indx)
            j=min(users_dict,key=lambda x:len(users_dict.get(x)))
            temp=set(np.random.choice(classes_indx,class_size,replace=False))
            classes_indx=list(set(classes_indx)-temp)
            for t in temp:
                users_dict[j]=np.concatenate((users_dict[j],indices[t*images:(t+1)*images]),axis=0)

    return users_dict



def load_dataset(num_users,iidtype):#this function helps load the datasets we made using mnistIID
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    trainset=torchvision.datasets.MNIST(root="./",train= False,transform=transform,download=False)
    testset=torchvision.datasets.MNIST(root="./",train= False,transform=transform,download=False)
    train_group=None
    test_group=None
    if iidtype=='iid':
        train_group=mnistIID(trainset,num_users)
        test_group=mnistIID(testset,num_users)
    elif iidtype=='noniid':
        train_group=mnistnonIID(trainset,num_users,True)
        test_group=mnistnonIID(testset,num_users,True)
    else:
        train_group=mnistnonIIDUnequal(trainset,num_users,True)
        test_group=mnistnonIIDUnequal(testset,num_users,True)
    return trainset,testset,train_group,test_group

class FedDataset(Dataset):#this class helps connect the random indices with the image+label container in the dataset
    def __init__(self,dataset,indx):
        self.dataset=dataset
        self.indx=[int(i) for i in indx]
        
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self,item):
        images,labels=self.dataset[self.indx[item]]
        return (torch.tensor(images),torch.tensor(labels))
    
    
def getImage(dataset,indices,batch_size):#load images using the class FedDataset
    return DataLoader(FedDataset(dataset,indices),batch_size=batch_size,shuffle=True)