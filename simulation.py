from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from collections import Counter,defaultdict


np.random.seed(42)
n_samples, n_features = 50,1611
no_clusters=8

fileName1="predicted"
fileName2="Actual"

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print(Counter(estimator.labels_))
    print(82*'_')
    for index,label in enumerate(estimator.labels_):
        clusters[label].append(predicted_vals[index])
    

for epoch in range(20):
	predicted_vals=[]
	actual_vals=[]
	for i in range(50):
		filename=fileName1+str(i+1)+".txt"
		Predicted=np.loadtxt(filename,dtype=int)
		predicted_vals.append(Predicted[epoch])
		filename=fileName2+str(i+1)+".txt"
		Actual=np.loadtxt(filename,dtype=int)
		actual_vals.append(Actual[epoch])
	print(len(predicted_vals),len(predicted_vals[0]))
	print(len(actual_vals),len(actual_vals[0]))
	predicted_vals=np.array(predicted_vals)
	data = scale(predicted_vals)
	clusters=defaultdict(list)
	#bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10), name="k-means++", data=data)
	#bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10), name="random", data=data)
	pca = PCA(n_components=no_clusters).fit(data)
	bench_k_means(KMeans(init=pca.components_, n_clusters=no_clusters, n_init=1),name="PCA-based",data=data)
	print(len(clusters))
	for key in clusters:
		print(82*'_')
		print(key)
		print(np.array(clusters[key]))
	quit()
	
	
	# reduced_data = PCA(n_components=2).fit_transform(data)
	# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
	# kmeans.fit(reduced_data)
	# h=0.02
	# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
	# Z = Z.reshape(xx.shape)
	# plt.figure(1)
	# plt.clf()
	# plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')
	# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
	# # Plot the centroids as a white X
	# centroids = kmeans.cluster_centers_
	# plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
	# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
			  # 'Centroids are marked with white cross')
	# plt.xlim(x_min, x_max)
	# plt.ylim(y_min, y_max)
	# plt.xticks(())
	# plt.yticks(())
	# plt.show()


