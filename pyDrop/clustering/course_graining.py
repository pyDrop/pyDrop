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
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class ModuloBins:
    """
    ModuloBins
    """
    def __init__(self, mod=10, rem=0):
        self.mod = mod
        self.rem = rem

        # Bin finding/evaluating functions to be called with np.arrays
        self.id_to_bin_start = np.vectorize(self._id_to_bin_start)
        self.id_to_bin_center = np.vectorize(self._id_to_bin_center)
        self.value_to_id = np.vectorize(self._value_to_id)

    def _id_to_bin_start(self, idx):
        return float(idx*self.mod + self.rem)
    
    def _id_to_bin_center(self, idx):
        return float(idx*self.mod + self.rem) + self.mod/2
    
    def _value_to_id(self, value):
        return int((value-self.rem)/self.mod)
    
class LinSpaceBins:
    """
    LinSpaceBins
    """
    def __init__(self, min, max, n_bins=100):
        self.min = min
        self.max = max
        self.n_bins = n_bins

        self.bins = np.linspace(self.min, self.max, self.n_bins)
        self.step = self.bins[1] - self.bins[0]

        # Bin finding/evaluating functions to be called with np.arrays
        self.id_to_bin_start = np.vectorize(self._id_to_bin_start)
        self.id_to_bin_center = np.vectorize(self._id_to_bin_center)
        self.value_to_id = np.vectorize(self._value_to_id)
    
    def id_to_bin_start(self, idx):
        return self.bins[idx]
    
    def id_to_bin_center(self, idx):
        return self.bins[idx] + self.step/2
    
    def value_to_id(self, value):
        return np.searchsorted(self.bins, value, side='right')
    
class ArrangeBins(LinSpaceBins):
    """
    ArrangeBins
    """
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step

        self.bins = np.arange(self.min, self.max, self.step)
        self.n_bins = len(self.bins)

        # Bin finding/evaluating functions to be called with np.arrays
        self.id_to_bin_start = np.vectorize(self._id_to_bin_start)
        self.id_to_bin_center = np.vectorize(self._id_to_bin_center)
        self.value_to_id = np.vectorize(self._value_to_id)

class Bins:
    """
    BinAxes: 
    """
    def __init__(self, default_binf=ModuloBins):
        self.default_binf = default_binf
        self.bin_functions = []

    def add_axis(self, binf=None):
        if not binf:
            binf = self.default_binf()
        self.bin_functions.append(binf)

    def get_binf(self, axis):
        return self.bin_functions[axis]
    
    def id_to_bin_start(self, ids):
        assert len(ids) == len(self.bin_functions)
        values = []
        for binf, idx in zip(self.bin_functions, ids):
            values.append(binf.id_to_bin_start(idx))
        return values
    
    def id_to_bin_center(self, ids):
        assert len(ids) == len(self.bin_functions)
        values = []
        for binf, idx in zip(self.bin_functions, ids):
            values.append(binf.id_to_bin_center(idx))
        return values
    
    def value_to_id(self, values):
        assert len(values) == len(self.bin_functions)
        ids = []
        for binf, value in zip(self.bin_functions, values):
            ids.append(binf.value_to_id(value))
        return ids

class KNNCalico:
    def __init__(self, X, n_clusters, bins=Bins, k_means_model=KMeans):
        """
        X: numpy array of size (n_samples, n_features)
        n_clusters: number of clusters to expect in data
        n_grids: number of grids along each dimension, defaults to 100 for every dimension
                 To have different numbers of grids along each axis, pass N-grids as a list
                    n_grids = [<n_grids for feature 1>, <n_grids for feature 2>, ...]
        """
        # initialize data and the expected number of clusters 
        self.X = X
        self.n_train, self.n_features = X.shape
        self.bins = bins()
        self.coarse_grained = False

        self.k_means_model = k_means_model(n_clusters)

    def fit_coarse_grain(self, bin_functions=None):
        if not bin_functions:
            bin_functions = [ModuloBins(10)]*self.n_features
        
        if len(bin_functions) != self.n_features:
            raise Exception("number of given bin functions must be equal to the number of features")
        
        for bin_function in bin_functions:
            self.bins.add_axis(binf=bin_function) 

        self.coarse_grained = True

    def fit_uniform_coarse_grain(self, binf=ModuloBins(10)):
        bin_functions = [binf]*self.n_features
        self.fit_coarse_grain(bin_functions)
    
    def fit(self, X):
        if not self.coarse_grained:
            self.fit_coarse_grain()
        
        # X_grained_bins = self.bins.value_to_id(X)
        # X_grained_centers = self.bins.id_to_bin_center(X_grained_bins)
        # self.k_means_model.fit(X_grained_centers)
        # centers = self.k_means_model._get the centers out
        # train new k_means model using new centers
        return

    def predict(self, X):
        return
    
    def score(X, y, type="rand"):
        return
    
if __name__ == "__main__":
    # from sklearn.datasets import make_blobs
    # import matplotlib.pyplot as plt
    # n_samples = 200
    # random_state = 170
    # n_blobs = 2

    # X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_blobs, random_state=random_state)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()

