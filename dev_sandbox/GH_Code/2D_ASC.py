import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score
from pathlib import Path
#dir and print functions/statements
# data_dir = Path("../../data")
# for file in data_dir.glob("**/X00*.csv"):
#     if file.exists() and file.is_file():
#         print(f"processing {file.name}")
#         df = pd.read_csv(file)
#         if "Ch2" not in df.columns:
#             print("Hey, this doesn't have Ch2 column!")
#         print(df.head())


# Load the dataset
df = pd.read_csv("../../data/X002_droplet_amplitudes.csv")

# Extract the features and true labels
features = df[['Ch1', 'Ch2']].values
true_labels = df["Cluster_1_2"].values

# Set the number of clusters
num_clusters = 4

# Define the clustering models
models = [    cluster.KMeans(n_clusters=num_clusters),    cluster.AgglomerativeClustering(n_clusters=num_clusters),    cluster.DBSCAN(),    cluster.OPTICS(),    cluster.Birch(n_clusters=num_clusters)]

# Calculate the evaluation metrics for each model
rand_scores = []
homogeneity_scores = []
completeness_scores = []
v_measure_scores = []

for model in models:
    # Fit the model and make predictions
    predicted_labels = model.fit_predict(features)

    # Calculate the evaluation metrics
    rand = rand_score(true_labels, predicted_labels)
    homo = homogeneity_score(true_labels, predicted_labels)
    comp = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    # Add the scores to the lists
    rand_scores.append(rand)
    homogeneity_scores.append(homo)
    completeness_scores.append(comp)
    v_measure_scores.append(v_measure)

# Plot the results
labels = ['KMeans', 'Agglomerative', 'DBSCAN', 'OPTICS', 'BIRCH']
x_axis = np.arange(len(labels))
bar_width = 0.2

plt.bar(x_axis - bar_width*1.5, rand_scores, bar_width, label='Rand Score')
plt.bar(x_axis - bar_width/2, homogeneity_scores, bar_width, label='Homogeneity Score')
plt.bar(x_axis + bar_width/2, completeness_scores, bar_width, label='Completeness Score')
plt.bar(x_axis + bar_width*1.5, v_measure_scores, bar_width, label='V-measure Score')

# Add the labels to the bars
for i in range(len(x_axis)):
    plt.text(x_axis[i] - bar_width*1.5, rand_scores[i] + 0.01, str(round(rand_scores[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(x_axis[i] - bar_width/2, homogeneity_scores[i] + 0.01, str(round(homogeneity_scores[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(x_axis[i] + bar_width/2, completeness_scores[i] + 0.01, str(round(completeness_scores[i], 2)), color='black', fontweight='bold', ha='center')
    plt.text(x_axis[i] + bar_width*1.5, v_measure_scores[i] + 0.01, str(round(v_measure_scores[i], 2)), color='black', fontweight='bold', ha='center')

plt.xticks(x_axis, labels)
plt.xlabel("Cluster")
plt.ylabel("Metric")
plt.title("Clustering Accuracy")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
