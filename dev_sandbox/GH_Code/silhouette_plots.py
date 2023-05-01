from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def silhouette(daf: str, mav: int, dims: Optional[int] = None) -> None:
    """
    Plot the silhouette scores for KMeans clustering on the specified data frame.

    Parameters:
    - daf (str): The file path to the data frame. Put filepath in double quotes
    - mav (int): The maximum number of clusters to test.
    - dims (int): The number of dimensions in the input data frame. Defaults to None.

    Returns:
    - The optimal number of clusters for a given data frame
    """
    # Check input values
    if not isinstance(mav, int) or mav <= 0:
        raise ValueError("mav must be a positive integer")
    if dims is not None and (not isinstance(dims, int) or dims <= 0):
        raise ValueError("dims must be a positive integer or None")

    # Import the data set
    try:
        df = pd.read_csv(daf)
    except FileNotFoundError:
        raise ValueError("File not found")

    # Extract data and true clustering values from dataframe
    if dims is None:
        features = df.values
    else:
        features = df.iloc[:, :dims].values

    # Set range of clusters to try
    k_values = range(2, mav + 1)

    # Initialize list to store silhouette scores
    silhouette_scores = []

    # Loop through cluster values and calculate silhouette score for each
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
        if score == max(silhouette_scores): #stores the optimal number of clusters as the maxk value
                maxk = []
                maxk = k

    print ('The optimal number of clusters is:', maxk)
    # Plot the silhouette scores
    plt.plot(k_values, silhouette_scores, 'bo-', color='b', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(f'Silhouette plot for KMeans clustering on {daf}')
    plt.show()


silhouette("../../data/3d_assay_6.csv", 12, 3)
