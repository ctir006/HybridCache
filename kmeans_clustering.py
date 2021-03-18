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
     