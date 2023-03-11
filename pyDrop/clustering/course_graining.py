"""
=========================================================================================
                                 CALICO CLUSTERING
=========================================================================================
(Iterative clustering assisted by coarse-graining) developed in the following paper
doi: 10.1021/acs.analchem.7b02688 
Purpose: Increases the ability of unsupervised models to capture smaller clusters through 
    course-graining. Iteratively course-grains the data the updated the starting location 
    of the K-means clustering algorithm
"""
import sklearn
import numpy as np

class Calico:
    def __init__(self, X, n_clusters, n_grids=100):
        """
        X: numpy array of size (n_samples, n_features)
        n_clusters: number of clusters to expect in data
        n_grids: number of grids along each dimension, defaults to 100 for every dimension
                 To have different numbers of grids along each axis, pass N-grids as a list
                    n_grids = [<n_grids for feature 1>, <n_grids for feature 2>, ...]
        """
        # initialize data and the expected number of clusters 
        self.X = X
        self.n_clusters = n_clusters

        # initialze number of grids
        if isinstance(n_grids, int):
            self.n_grids = [n_grids]*self.n_clusters
        elif isinstance(n_grids, list):
            assert len(n_grids) == self.n_clusters
            self.n_grids = n_grids
        elif isinstance(n_grids, np.ndarray):
            n_grids = n_grids.to_list()
            assert len(n_grids) == self.n_clusters
            self.n_grids = n_grids
        else:
            Exception(f"unsupported type {type(n_grids)} for n_grids")

        self.n_grids = n_grids
        self.k_means_model = sklearn.cluster.KMeans

    def fit(X):
        return

    def predict(X):
        return
    
    def _reorder_predictions(X, y_hat):
        return
    
    def _create_grid():
        return 
    
    def _course_grain():
        return