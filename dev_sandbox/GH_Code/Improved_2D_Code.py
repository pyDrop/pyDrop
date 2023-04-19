import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score

def clustering_accuracy(df,clusters=None):
    #Extracting data and true clustering values from dataframe
    features = df.iloc[:,:-2].values
    true = df.iloc[:,-2].values

    #Defining default cluster value as a function of channels
    if clusters == None:
        clusters = features.shape[1]*2

    #Creating an array of existing clustering algorithms
    models = [cluster.KMeans(n_clusters=clusters), cluster.AgglomerativeClustering(n_clusters=clusters),
              cluster.DBSCAN(), cluster.OPTICS(), cluster.Birch(n_clusters=clusters)]
    results = []

    #Determing accuracy of each clustering model with
    for model in models:
        predictions = model.fit_predict(features)
        # Creating an array of accuracy scores
        scores = [rand_score(true, predictions), homogeneity_score(true, predictions),
                  completeness_score(true, predictions), v_measure_score(true, predictions)]
        results.append(scores)

    #Making results vector the correct shape
    results = np.transpose(results)

    #Creating Bar Graph
    labels = ['KMeans', 'AggClus', 'DBSCAN', 'OPTICS', 'Birch']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2 * width, results[0], width, label='Rand')
    rects2 = ax.bar(x - width, results[1], width, label='Homogeneity')
    rects3 = ax.bar(x, results[2], width, label='Completeness')
    rects4 = ax.bar(x + width, results[3], width, label='V-measure')

    # Add some text for column values centered above each column
    for rect in rects1 + rects2 + rects3 + rects4:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=6)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Clustering Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower left', fontsize=10)

    plt.show()

df = pandas.read_csv("../../data/X002_droplet_amplitudes.csv")
clustering_accuracy(df)
