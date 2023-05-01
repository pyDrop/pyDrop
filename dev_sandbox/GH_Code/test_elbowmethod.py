import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_method(df):

    # Defining features, true cluster values, and amount of clusters
    if 'droplet' in df.columns:
        df = df.drop('droplet', axis=1)

    # Extracting data and true clustering values from dataframe
    features = df.iloc[:, :-1].values

    # Run KMeans with different number of clusters
    k_values = range(1, 10)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Calculate the slopes between adjacent points
    slopes = np.diff(inertias) / np.diff(k_values)

    # Find the index of the point with the largest change in slopes
    largest_change_index = np.argmax(np.abs(np.diff(slopes))) + 1

    # Calculate the largest change in slopes
    largest_change = np.abs(slopes[largest_change_index] - slopes[largest_change_index - 1])

    # Plot the elbow graph
    fig, ax = plt.subplots()
    ax.plot(k_values, inertias, 'bo-')
    ax.set_xlabel('K-value')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')

    # Add text to the plot
    textstr = 'K-value where the largest change in slope occurs: k={}'.format(k_values[largest_change_index])
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    #print statement to see the where the largest change in slope occurs
    print("The largest change in slope occurs at a k value of:", largest_change_index)

    plt.show()

df1 = pd.read_csv("../../data/X001_droplet_amplitudes.csv")
df2 = pd.read_csv("../../data/X007_droplet_amplitudes.csv")
df3 = pd.read_csv("../../data/3d_assay_4.csv")
elbow_method(df2)
