import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score

df = pd.read_csv("../../data/X001_droplet_amplitudes.csv")
goat = df['Ch1'].to_numpy().reshape(-1, 1)
true = df["Cluster_1"].to_numpy()
clusters = 2
models = [cluster.KMeans(n_clusters=clusters), cluster.AgglomerativeClustering(n_clusters=clusters),
          cluster.DBSCAN(), cluster.OPTICS(), cluster.Birch(n_clusters=clusters)]

rand = []
homo = []
comp = []
vmeasure = []

for model in models:
    predictions = model.fit_predict(goat)
    rand_temp = rand_score(true, predictions)
    homo_temp = homogeneity_score(true, predictions)
    comp_temp = completeness_score(true, predictions)
    vmeasure_temp = v_measure_score(true, predictions)
    rand.append(rand_temp)
    homo.append(homo_temp)
    comp.append(comp_temp)
    vmeasure.append(vmeasure_temp)

labels = ['KMeans', 'Agglomerative', 'DBSCAN', 'OPTICS', 'BIRCH']
X_axis = np.arange(len(labels))

bar_width = 0.2
plt.bar(X_axis - bar_width*1.5, rand, bar_width, label='Rand Score')
plt.bar(X_axis - bar_width/2, homo, bar_width, label='Homogeneity Score')
plt.bar(X_axis + bar_width/2, comp, bar_width, label='Completeness Score')
plt.bar(X_axis + bar_width*1.5, vmeasure, bar_width, label='V-measure Score')

# Add the labels to the bars
for i in range(len(X_axis)):
    plt.text(X_axis[i] - bar_width*1.5, rand[i] + 0.01, str(round(rand[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(X_axis[i] - bar_width/2, homo[i] + 0.01, str(round(homo[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(X_axis[i] + bar_width/2, comp[i] + 0.01, str(round(comp[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(X_axis[i] + bar_width*1.5, vmeasure[i] + 0.01, str(round(vmeasure[i], 2)), color='black', fontweight='bold', ha='center')

plt.xticks(X_axis, labels)
plt.xlabel("Cluster")
plt.ylabel("Metric")
plt.title("Clustering Accuracy")
plt.legend(loc='lower right')
plt.show()