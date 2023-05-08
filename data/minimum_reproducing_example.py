import pandas as pd
import numpy as np

# Read the data using pandas and convert to numpy
data = pd.read_csv("3d_assay_2.csv", sep=",", header=0)
X = data[["Ch1", "Ch2", "Ch3"]].to_numpy() # features/X data/3D coords
y = data[["cluster"]].to_numpy() # cluster assignments

# Convert y from format of 100000, 110000, 101000, etc to 0, 1, 2, etc. 
labels = np.unique(y) # find unique labels
label_to_cluster_id_dict = dict([(label, idx) for idx, label in enumerate(labels)]) # map labels to ids
label_to_cluster_id = np.vectorize(lambda x: label_to_cluster_id_dict[x]) # vectorize to numpy can use on an entire array at the same time
y = label_to_cluster_id(y) # convert

print(f"X data is a {type(X)} of shape {X.shape}")
print(X[:5, :])
print(f"Xydata is a {type(y)} of shape {y.shape}")
print(y[:5])

# vvvvvvvvvvvv YOUR CODE HERE vvvvvvvvvvvvvvvv


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# you can put your functions here or import them
# from another file. 