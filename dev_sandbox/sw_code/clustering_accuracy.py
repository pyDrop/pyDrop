import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score

def clustering_accuracy(df, dimensions=None):

    #Defining features, true cluster values, and amount of clusters
    if 'droplet' in df.columns:
        # Extracting data and true clustering values from dataframe
        features = df.iloc[:, :-2].values
        true = df.iloc[:, -2].values
        # Finds how many unique cluster names
        clusters = len(df.iloc[:, -2].unique())
    else:
        # Extracting data and true clustering values from dataframe
        features = df.iloc[:, :-1].values
        true = df.iloc[:, -1].values
        # Finds how many unique cluster names
        clusters = len(df.iloc[:, -1].unique())

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

    fig1, ax1 = plt.subplots()
    rects1 = ax1.bar(x - 2 * width, results[0], width, label='Rand')
    rects2 = ax1.bar(x - width, results[1], width, label='Homogeneity')
    rects3 = ax1.bar(x, results[2], width, label='Completeness')
    rects4 = ax1.bar(x + width, results[3], width, label='V-measure')

    # Add some text for column values centered above each column
    for rect in rects1 + rects2 + rects3 + rects4:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Score')
    ax1.set_title('Clustering Evaluation Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='lower right', fontsize=10)



    # List of unique hexcode colors
    colors = ['#0000FF', '#000000', '#FF0000', '#FFC0CB', '#FFA07A', '#FF7F50',
              '#FF4500', '#FFD700', '#FFFF00', '#00FF00', '#32CD32', '#00FFFF',
              '#008080', '#000080', '#8B00FF', '#FF69B4', '#BA55D3', '#800080',
              '#FF6347', '#CD5C5C', '#A0522D', '#696969']
    
    # Create 1,2 or 3 dimensional scatter plot
    if dimensions == 1:
        clusters = df.iloc[:, 1].unique()  # Use integer position 1 instead of column name

        fig2, ax2 = plt.subplots()  # Create a new figure and axis for the 1D plot

        for index, value in enumerate(clusters):
            droplet = (df[df.iloc[:, 1] == value].iloc[:, -1].to_numpy())  # find droplet value for given cluster
            channel = (df[df.iloc[:, 1] == value].iloc[:, 0].to_numpy())  # find channel value for given cluster
            ax2.scatter(droplet, channel, color=colors[index])  # applies unique color to cluster

        # set axis labels and title
        ax2.set_xlabel('Droplet')
        ax2.set_ylabel('Channel')
        ax2.set_title('1D Clustering')

    if dimensions == 2:
        clusters = df.iloc[:, 2].unique()

        fig2, ax2 = plt.subplots()  # Create a new figure and axis for the 2D plot

        for index, value in enumerate(clusters):
            channel1 = (df[df.iloc[:, 2] == value].iloc[:, 0].to_numpy())  # find channel1 value for given cluster
            channel2 = (df[df.iloc[:, 2] == value].iloc[:, 1].to_numpy())  # find channel2 value for given cluster
            ax2.scatter(channel1, channel2, color=colors[index])  # applies unique color to cluster

        # set axis labels and title
        ax2.set_xlabel('Channel 1')
        ax2.set_ylabel('Channel 2')
        ax2.set_title('2D Clustering')

    if dimensions == 3:
        # get the unique clusters and set the color scheme
        clusters = df.iloc[:, -1].unique()

        fig2 = plt.figure()  # Create a new figure for the 3D plot
        ax2 = fig2.add_subplot(111, projection='3d')

        # iterate over the clusters and add each cluster's data to the same axis object
        for index, value in enumerate(clusters):
            channel1 = df.loc[df.iloc[:, -1] == value, df.columns[0]]
            channel2 = df.loc[df.iloc[:, -1] == value, df.columns[1]]
            channel3 = df.loc[df.iloc[:, -1] == value, df.columns[2]]
            ax2.scatter(channel1, channel2, channel3, c=colors[index])

        # set labels for the axes
        ax2.set_xlabel('Channel 1')
        ax2.set_ylabel('Channel 2')
        ax2.set_zlabel('Channel 3')
        ax2.set_title('3D Clustering')

    plt.show()

# df1 = pd.read_csv("../../data/X001_droplet_amplitudes.csv")
# df2 = pd.read_csv("../../data/X007_droplet_amplitudes.csv")
# df3 = pd.read_csv("../../data/3d_assay_4.csv")
# clustering_accuracy(df3,3)