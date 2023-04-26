import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



# Initialize an empty list to store the inertia values
inertia = []

# Try different values of k from 1 to 10 and calculate the inertia for each value of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#function code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_method(X, kmax):
    """
    Implement the elbow method for finding the optimal number of clusters in K-means clustering.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The data to cluster.
    kmax : int
        The maximum number of clusters to try.

    Returns:
    None
    """
    # Initialize an empty list to store the inertia values
    inertia = []

    # Try different values of k from 1 to kmax and calculate the inertia for each value of k
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, kmax+1), inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()


#df = pd.read_csv("../../data/X002_droplet_amplitudes.csv")
