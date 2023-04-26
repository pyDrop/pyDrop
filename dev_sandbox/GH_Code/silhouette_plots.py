from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def silhouette(df, ma, dimensions=None)
# df - specifying the data frame
# ma - specifying the maximum number of clusters to test
# dimensions - used to determine the dimensions of the input data fram



    # Extracting data and true clustering values from dataframe
assert isinstance(df.iloc[:, dimensions - 1].values, object)
features = df.iloc[:, dimensions - 1].values

#

    

# # Generate sample data
# X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.6, random_state=0)
#
# # Define range of k values to evaluate
# range_n_clusters = [2, 3, 4, 5, 6]
#
# for n_clusters in range_n_clusters:
#     # Initialize KMeans
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#
#     # Compute silhouette scores for each sample
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     # Plot the silhouette plot for each cluster
#     fig, ax = plt.subplots(1, 1)
#     fig.set_size_inches(8, 4)
#     ax.set_xlim([-0.1, 1])
#     ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to cluster i
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#
#         # Sort the silhouette scores
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = plt.cm.nipy_spectral(float(i) / n_clusters)
#         ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10
#
#     ax.set_title("Silhouette plot for KMeans clustering with n_clusters = %d" % n_clusters)
#     ax.set_xlabel("Silhouette coefficient values")
#     ax.set_ylabel("Cluster labels")
#
#     # Draw a vertical line at the average silhouette score
#     ax.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     plt.show()
