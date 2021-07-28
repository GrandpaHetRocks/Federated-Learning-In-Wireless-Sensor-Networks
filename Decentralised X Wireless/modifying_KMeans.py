# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
import pandas as pd
import math
import numpy as np
import random
# define dataset
cluster_array, _ = make_classification(n_samples=30, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=2)

no=1
clients={}
for client in cluster_array:
    clients['client'+str(no)]=client
    no+=1

def get_key(val,my_dict):
    for key, value in my_dict.items():
         if (val[0] == value[0] and val[1] == value[1]):
             return key

def calc_distance(X1, X2):
    return (sum((X1 - X2)**2))**0.5

# Assign cluster clusters based on closest centroid
def assign_clusters(centroids, cluster_array,clients,snr_list):
    clusters = []
    cluster_head=[]
    mindis1=10000000
    mindis2=10000000
    cluster_head1=''
    cluster_head2=''
    snr1=0
    snr2=0
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid,cluster_array[i]))
        #print(distances)
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        #clusters.append(cluster[0])
        if(min(distances)<=mindis1 and cluster[0]==0):
            cluster_head1=get_key(cluster_array[i],clients)
            mindis1=min(distances)
        if(min(distances)<=mindis2 and cluster[0]==1):
            cluster_head2=get_key(cluster_array[i],clients)
            mindis2=min(distances)
        #print(60*math.exp(-min(distances))+random.uniform(-10,10))
        for m in snr_list:
            if(cluster_head1 in m and get_key(cluster_array[i],clients) in m):
                snr1=m[2]
            if(cluster_head2 in m and get_key(cluster_array[i],clients) in m):
                snr2=m[2]
                break
        #print(snr1,snr2)
        if(snr1-distances[0]>=snr2-distances[1]):
            #clusters.pop()
            clusters.append(0)
        else:
            #clusters.pop()
            clusters.append(1)
            #print(snr1,snr2)
        
    
    #print(cluster_head1,cluster_head2)
    return (clusters,cluster_head1,cluster_head2)

# Calculate new centroids based on each cluster's mean
def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters,columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

def snr_calc(clients):
    snr_list=[]
    for i in range(len(clients)-1):
        for j in range(i+1,len(clients)):
            X1=clients['client'+str(i+1)]
            X2=clients['client'+str(j+1)]
            dis=calc_distance(X1,X2)
            snr=60*math.exp(-dis)+random.uniform(-5,5)
            snr_list.append(['client'+str(i+1),'client'+str(j+1),snr])
    return(snr_list)

snr=snr_calc(clients)
#print(snr)

k = 2
cluster_vars = []
centroids = [cluster_array[i+2] for i in range(k)]
clusters,cluster_head1,cluster_head2  = assign_clusters(centroids, cluster_array,clients,snr)
initial_clusters = clusters



for i in range(20):
    centroids = calc_centroids(clusters, cluster_array)
    clusters,cluster_head1,cluster_head2 = assign_clusters(centroids,cluster_array,clients,snr)



#print(len(clusters))
x=[k[0] for k in cluster_array]
#print(x)
y=[k[1] for k in cluster_array]
for i in range(30):
    if(clusters[i]==0):
        color='blue'
    else:
        color='red'
    pyplot.scatter(x[i],y[i],c=color)


ch1=clients[cluster_head1]
ch2=clients[cluster_head2]

pyplot.scatter(ch1[0],ch1[1],c='green')
pyplot.scatter(ch2[0],ch2[1],c='green')
pyplot.show()


