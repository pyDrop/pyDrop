from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def silhouette(daf, mav, dims=None):
    #daf - specifying the file path to the data frame
    #ex daf = "../../data/X002_droplet_amplitudes.csv", must be in quotes
    # mav - specifying the maximum number of clusters to test
    # dimensions - used to determine the dimensions of the input data frame

    #import the data set
    df = pd.read_csv(daf)

    # Extracting data and true clustering values from dataframe

    features = df.iloc[:, dims - 1].values

    #definition a matrix of cluster values
    k_values = range(1, mav)

for n_clusters in range_n_clusters:

    #initialize K_Means
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(features)

    #Compute the silhouette scores for each sample
    silhouette_avg = silhouette_score(daf , cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


    
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
#
# # Generate sample data
# X, y = make_blobs(n_samples=1000, centers=8, n_features=2, random_state=42)
#
# # Set range of clusters to try
# k_values = range(2, 11)
#
# # Initialize list to store silhouette scores
# silhouette_scores = []
#
# # Loop through cluster values and calculate silhouette score for each
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     score = silhouette_score(X, kmeans.labels_)
#     silhouette_scores.append(score)
#
# # Plot the silhouette scores
# plt.plot(k_values, silhouette_scores, 'bo-', color='b', linewidth=2, markersize=8)
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.title('Silhouette plot for KMeans clustering')
# plt.show()

