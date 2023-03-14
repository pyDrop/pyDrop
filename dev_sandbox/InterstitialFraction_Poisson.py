import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from scipy.stats import nbinom, poisson, geom
from sklearn.cluster import KMeans

dframe = pd.read_csv("../data/X001_droplet_amplitudes.csv", header=0)
cluster0 = dframe[dframe['Cluster_1'] == 0]['Ch1'].to_numpy()  # create numpy array for calculations
cluster1 = dframe[dframe['Cluster_1'] == 1]['Ch1'].to_numpy()

# General approach is as follows:
# Calculate the parameters for a given distribution using method of moments
# Using point percent function, get wavelengths to filter the distribution
# Determine the fraction of points that don't fall between these values


# Summary statistics for each cluster Poisson - single parameter, the mean
lambda0 = cluster0.mean()
lambda1 = cluster1.mean()

# Using the percent point function, get the exact wavelength where it becomes probable that a datapoint is out of range
# Alpha is the probability corresponding to "out of range" arbitrarily small choice for now
alpha = 0.01

# For channel zero we seek the wavelelngth where CDF(X<x) = 1 - alpha (above the range)
bound_cluster0 = poisson.ppf(1-alpha, lambda0)
# For channel one, we seek the wavelength where CDF(X<x) = alpha (below the range)
bound_cluster1 = poisson.ppf(alpha, lambda1)

# By converting each cluster vector into a logical vector and summing it, we determine the fraction of data points that are out of spec
interstitialFraction = ((cluster1<bound_cluster1).sum()+(cluster0>bound_cluster0).sum())/(cluster1.size+cluster0.size)
interstitialFraction0 = (cluster0>bound_cluster0).sum()/(cluster1.size+cluster0.size)
interstitialFraction1 = (cluster1<bound_cluster1).sum()/(cluster1.size+cluster0.size)


print("The upper bound on cluster 0 is: ", bound_cluster0, "and the lower bound on cluster 1 is: ", bound_cluster1)
print(f"The interstitial fraction of the data is:  {interstitialFraction*100:.2f} %")
print(f"Of the interstitial points, {interstitialFraction0/interstitialFraction*100:.2f} % are contributed by cluster0 and {interstitialFraction1/interstitialFraction*100:.2f} % are contributed by cluster1")





