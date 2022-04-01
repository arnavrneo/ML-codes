import numpy as np
import pandas as pd

from sklearn.cluster import KMeans # the KNN Clustering module
from sklearn.datasets import make_blobs # to create artificial data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('talk')
plt.rcParams['figure.figsize'] = [10, 8]

# Example 1: clustering dotted circle
angles = np.linspace(0, 2*np.pi, 20)
x = np.append([np.cos(angles)], [np.sin(angles)], 0).transpose()

# function for creating the plot
def plot(x, km=[], num_clusters=0):
    color = 'brgmyck'
    s=20
    if num_clusters == 0:
        plt.scatter(x[:, 0], x[:, 1],c=color[0], s=s)
    else:
        for i in range(num_clusters):
             plt.scatter(x[km.labels_==i,0],x[km.labels_==i,1],c = color[i], s=s)
             plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)

# trying out KMeans on the dotted circle
clusters = 2
km = KMeans(n_clusters=clusters, random_state=101)
km.fit(x)
plot(x, km, clusters)


# Example 2: artificial data
samples = 1000
centers = [(-3, -3), (0, 0), (3, 3), (6, 6)]
x, y = make_blobs(n_samples=samples, n_features=2,
                 centers=centers, shuffle=False, random_state=101)
plot(x)

# trying out KMeans on the artificial data
clusters = 6
km = KMeans(n_clusters=clusters)
km.fit(x)
plot(x, km, clusters)

# Choosing the value of K
inertia = []
list_clusters = list(range(1,11))
for num_clusters in list_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(x)
    inertia.append(km.inertia_)
    
plt.plot(list_clusters,inertia)
plt.scatter(list_clusters,inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')


# Example 3: Image colors clustering
img = plt.imread('ironman2.jpg') # reading an image (image address here)
plt.imshow(img)
plt.axis('off')

img.shape

img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
img_flat

img_flat.shape

kmeans = KMeans(n_clusters=8, random_state=0).fit(img_flat)

img_flat2 = img_flat.copy()

# loops for each cluster center
for i in np.unique(kmeans.labels_):
    img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]

img2 = img_flat2.reshape(img.shape)
plt.imshow(img2)
plt.axis('off')

def img_cluster(img, k):
    img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    img_flat2 = img_flat.copy()
    
    for i in np.unique(kmeans.labels_):
        img_flat2[kmeans.labels_==i, :] = kmeans.cluster_centers_[i]
        
    img2 = img_flat2.reshape(img.shape)
    return img2, kmeans.inertia_


plt.figure(figsize=[10,20])
for i in range(len(k_vals)):
    plt.subplot(5,2,i+1)
    plt.imshow(img_list[i])
    plt.title('k = '+ str(k_vals[i]))
    plt.axis('off')
