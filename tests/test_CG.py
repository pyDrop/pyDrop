"""
pytest script to test coarse-graining, binning, and KNNCalico
"""
import pytest

from sklearn.datasets import make_blobs
from pyDrop.clustering import *

def test_modulo_bins():
    binf1 = ModuloBins(mod=100, rem=0)
    binf2 = ModuloBins(mod=25, rem=10)
    assert binf1._id_to_bin_start(1) == 100
    assert binf1._id_to_bin_center(1) == 150
    assert binf1._value_to_id(1234) == 12

    assert binf2._id_to_bin_start(1) == 35
    assert binf2._id_to_bin_center(1) == 47.5
    assert binf2._value_to_id(61) == 2

def test_linspace_bins():
    binf = LinSpaceBins(12, 113, 100)
    assert binf._id_to_bin_start(0) == 12
    assert binf._id_to_bin_start(-2) == 12
    assert binf._id_to_bin_start(100) == 111.99
    assert binf._id_to_bin_start(101) == 111.99
    assert binf._id_to_bin_center(101) == 112.495
    assert binf._value_to_id(65) == 52
    assert binf._value_to_id(12) == 0
    assert binf._value_to_id(10) == 0
    assert binf._value_to_id(114) == 100

def test_arrange_bins():
    binf = ArrangeBins(12,113,101/100)
    assert binf._id_to_bin_start(0) == 12
    assert binf._id_to_bin_start(-2) == 12
    assert binf._id_to_bin_start(100) == 111.99
    assert binf._id_to_bin_start(101) == 111.99
    assert binf._id_to_bin_center(101) == 112.495
    assert binf._value_to_id(65) == 52
    assert binf._value_to_id(12) == 0
    assert binf._value_to_id(10) == 0
    assert binf._value_to_id(114) == 100

def test_multibins():
    bins = Bins()
    bins.add_axis(ModuloBins(100, 0))
    bins.add_axis(LinSpaceBins(12, 113, 110))
    bins.add_axis(ArrangeBins(12, 113, 1))
    ids = np.array([[0,1,1],
                    [3,2,2],
                    [5,3,3],
                    [90,24,24],
                    [43,65,65],
                    [13,108,100]])
    starts = bins.id_to_bin_start(ids)
    centers = bins.id_to_bin_center(ids)
    cids = bins.value_to_id(centers)
    assert np.isclose(ids, cids).all()

    cstarts = np.array([[0.,12.91818182,13.],
                        [300.,13.83636364,14.],
                        [500.,14.75454545,15.],
                        [9000.,34.03636364,36.],
                        [4300.,71.68181818,77.],
                        [1300.,111.16363636,112.]])
    assert np.isclose(starts, cstarts).all()

def test_errors():
    n_samples = np.array([25000, 25000, 100])
    n_features = 2
    centers = np.array([[-3000, -3000],[-3000, 3000],[0,0]])
    stdevs = np.array([500, 500, 500])
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=stdevs)
    model = KMCalico(k_means_model=KMeans(n_clusters=3))
    model.fit(X)
    X_grained_centers = model.coarse_grain(X)
    with pytest.raises(ValueError):
        model.predict(X, model="some type")