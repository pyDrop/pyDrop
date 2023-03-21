import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score

df = pd.read_csv("../../data/X001_droplet_amplitudes.csv")
goat = df['Ch1'].to_numpy().reshape(-1, 1)
true = df["Cluster_1"].to_numpy()
