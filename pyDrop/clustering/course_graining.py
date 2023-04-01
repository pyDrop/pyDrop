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
from sklearn.cluster import KMeans, OPTICS, DBSCAN, Birch, AgglomerativeClustering
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pyDrop.utils import datasets

class ModuloBins:
    """
    ModuloBins
    """
    def __init__(self, mod=100, rem=0):
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

    def add_axes(self, binfs):
        for binf in binfs:
            self.add_axis(binf)

    def get_num_axes(self):
        return len(self.bin_functions)

    def get_binf(self, axis):
        return self.bin_functions[axis]
    
    #TODO: reduce the following functions a bit
    def id_to_bin_start(self, ids):
        if len(ids.shape) == 1:
            ids = ids.reshape(-1, 1)
        n_samples, n_features = ids.shape
        assert n_features == len(self.bin_functions)
        values = np.zeros(ids.shape)
        for binf, axis_idx in zip(self.bin_functions, range(n_features)):
            values[:,axis_idx] = binf.id_to_bin_start(ids[:,axis_idx])
        return values
    
    def id_to_bin_center(self, ids):
        if len(ids.shape) == 1:
            ids = ids.reshape(-1, 1)
        n_samples, n_features = ids.shape
        assert n_features == len(self.bin_functions)
        values = np.zeros(ids.shape)
        for binf, axis_idx in zip(self.bin_functions, range(n_features)):
            values[:,axis_idx] = binf.id_to_bin_center(ids[:,axis_idx])
        return values
    
    def value_to_id(self, values):
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        n_samples, n_features = values.shape
        assert n_features == len(self.bin_functions)
        ids = np.zeros(values.shape)
        for binf, axis_idx in zip(self.bin_functions, range(n_features)):
            ids[:,axis_idx] = binf.value_to_id(values[:,axis_idx])
        return ids

class CGCluster:
    def __init__(self, bins=Bins(), model=OPTICS()):
        self.bins = bins
        self.model = model

        self.coarse_model = deepcopy(model)

    def fit_uniform_coarse_grain(self, n_features, binf=ModuloBins(100)):
        bin_functions = [binf]*n_features
        self.bins.add_axes(bin_functions)

    def coarse_grain(self, X):
        n_samples, n_features = X.shape
        n_defined_bins = self.bins.get_num_axes()
        if n_defined_bins == 0:
            print(f"no axes defined, using default ModuloBins bins for coarse-graining")
            self.fit_uniform_coarse_grain(n_features)
        elif n_defined_bins < n_features:
            print(f"ambiguous bins definition: {n_defined_bins} axes defined for bins but\
                  there are {n_features} total axes.")
        
        X_grained_bins = self.bins.value_to_id(X)
        X_grained_bins = np.unique(X_grained_bins, axis=0)
        X_grained_centers = self.bins.id_to_bin_center(X_grained_bins)

        return X_grained_centers
    
    def fit(self, X):
        X_grained_centers = self.coarse_grain(X)
        self.coarse_model.fit(X_grained_centers)

    def predict(self, X, model="coarse"):
        y_pred = None
        if model=="coarse":
            y_pred = self.coarse_model.predict(X)
        elif model=="default":
            y_pred = self.model.fit_predict(X)
        else:
            raise Exception("model must be coarse or default")
        return y_pred
    
    def scores(self, X, y_true, model="fine"):
        y_pred = self.predict(X, model=model)
        r_score = rand_score(y_true, y_pred)
        scores = {"rand_score": r_score,
                  "homogeneity_score": homogeneity_score(y_true, y_pred),
                  "completeness_score": completeness_score(y_true, y_pred),
                  "n_misclassified": len(y_true)*(1-r_score)
                 }
        return scores

class KNNCalico(CGCluster):
    def __init__(self, bins=Bins(), k_means_model=KMeans()):
        """

        """
        # initialize data and the expected number of clusters 
        self.bins = bins
        self.model = k_means_model

        self.coarse_model = deepcopy(k_means_model)
        self.coarse_model.set_params(n_init='auto')

        self.fine_model = deepcopy(k_means_model)
        self.fine_model.set_params(n_init=1)

    def fit(self, X):
        X_grained_centers = self.coarse_grain(X)
        self.coarse_model.fit(X_grained_centers)
        coarse_centers = self.coarse_model.cluster_centers_

        self.fine_model.set_params(init=coarse_centers)
        self.fine_model.fit(X)

    def predict(self, X, model="fine", return_centers=False):
        y_pred = None
        centers = None
        if model=="fine":
            y_pred = self.fine_model.predict(X)
            centers = self.fine_model.cluster_centers_
        elif model=="coarse":
            y_pred = self.coarse_model.predict(X)
            centers = self.coarse_model.cluster_centers_
        elif model=="default":
            y_pred = self.model.fit_predict(X)
            centers = self.model.cluster_centers_
        else:
            raise Exception("model must be fine or coarse")
        if return_centers:
            return y_pred, centers
        else:
            return y_pred
    
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # X2, y2 = make_blobs(n_samples=50000, n_features=2, centers=np.array([[-3000, -3000],[-3000, 3000]]), cluster_std=250)
    # X3, y3 = make_blobs(n_samples = 120, n_features=2, centers=np.array([[0,0]]), cluster_std=250)
    # X = np.concatenate((X2, X3), axis=0)
    # model = KNNCalico(3)
    # model.fit(X)
    # plt.scatter(X[:,0], X[:,1], alpha=0.25)
    # X_grained_centers = model.coarse_grain(X)
    # plt.scatter(X_grained_centers[:,0], X_grained_centers[:,1], alpha=0.50)
    # for m in ["fine", "coarse", "knn"]:
    #     y_pred, centers = model.predict(X, model=m, return_centers=True)
    #     print(centers)
    #     plt.scatter(centers[:,0], centers[:,1], alpha=1, s=100, marker="x", label=m)
    # plt.legend()
    # plt.show()

    datasets.hithere()

