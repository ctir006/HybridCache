import numpy as np
from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter,defaultdict


class kmeans_clustering:
    def __init__(self,inputData,no_clusters):
        self.data=np.array(inputData)
        self.number_of_k=no_clusters
        
    def kmeans(self):
        scaler = MinMaxScaler()
        scaledData = self.data
        kmeans = KMeans(n_clusters=self.number_of_k, init='k-means++', verbose=0, random_state=3425).fit(scaledData)
        #print(kmeans.labels_)
        ClusterCounts = Counter(kmeans.labels_)
        Clusters = defaultdict(lambda:[])
        ClusterData = defaultdict(lambda:[])
        for index,val in enumerate(kmeans.labels_):
            Clusters[val].append(index)
            ClusterData[val].append(self.data[index])
        return Clusters,kmeans.labels_




'''
    Code to find number of clusters
    
        x=[]
        y=[]
        for noclusters in range(1,15):
            x.append(noclusters)
            kmeans = KMeans(n_clusters=noclusters).fit(scaledData)
            y.append(kmeans.inertia_)
            print(kmeans.inertia_)
        plt.figure()
        plt.plot(x,y)
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()
        quit()
        print(len(self.data),len(self.data[0]))
'''        