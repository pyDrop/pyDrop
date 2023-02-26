import matplotlib.pyplot as plt
import pandas as pd


rel_data_path = "data/X001_droplet_amplitudes.csv" # make sure you are in the very top directory of the project! 
dframe = pd.read_csv(rel_data_path, header=0)
print(dframe.head()) # take a look at the data
color_values = {0: 'red', 1: 'blue'} # define colors: for Cluster 0, make it red. For Cluster 1, make it blue
colors = [color_values[v] for v in dframe["Cluster_1"]] # create a list of colors using our color values
plt.scatter(dframe["droplet"], dframe["Ch1"], s=0.5, c=colors) # make the plot with colors
plt.xlabel("droplet number")
plt.ylabel("Channel 1 Reading")
plt.show()