import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from pyDrop.clustering import KMCalico

n_samples = np.array([25000, 25000, 100])
n_features = 2
centers = np.array([[-3000, -3000],[-3000, 3000],[0,0]])
stdevs = np.array([500, 500, 500])
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=stdevs)
model = KMCalico(k_means_model=KMeans(n_clusters=3))
model.fit(X)
plt.scatter(X[:,0], X[:,1], alpha=0.25)
X_grained_centers = model.coarse_grain(X)
plt.scatter(X_grained_centers[:,0], X_grained_centers[:,1], alpha=0.50)
for m in ["fine", "coarse", "default"]:
    y_pred, centers = model.predict(X, model=m, return_centers=True)
    print(centers)
    plt.scatter(centers[:,0], centers[:,1], alpha=1, s=200, marker="x", label=m)
plt.legend()
plt.show()
