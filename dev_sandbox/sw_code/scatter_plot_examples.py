import pandas as pd
import matplotlib.pyplot as plt

# List of unique hexcode colors
colors = ['#0000FF', '#000000', '#FF0000', '#FFC0CB', '#FFA07A', '#FF7F50',
          '#FF4500', '#FFD700', '#FFFF00', '#00FF00', '#32CD32', '#00FFFF',
          '#008080', '#000080', '#8B00FF', '#FF69B4', '#BA55D3', '#800080',
          '#FF6347', '#CD5C5C', '#A0522D', '#696969']


# 1D Data
df = pd.read_csv("../../data/X001_droplet_amplitudes.csv")
clusters = df.iloc[:, 1].unique()  # Use integer position 1 instead of column name

fig1, ax1 = plt.subplots()  # Create a new figure and axis for the 1D plot

for index, value in enumerate(clusters):
    droplet = (df[df.iloc[:, 1] == value].iloc[:, -1].to_numpy()) # find droplet value for given cluster
    channel = (df[df.iloc[:, 1] == value].iloc[:, 0].to_numpy())  # find channel value for given cluster
    ax1.scatter(droplet, channel, color=colors[index]) # applies unique color to cluster

# set axis labels and title
ax1.set_xlabel('Droplet')
ax1.set_ylabel('Channel')
ax1.set_title('1D Clustering')



# 2D Data
df = pd.read_csv("../../data/X007_droplet_amplitudes.csv")
clusters = df.iloc[:, 2].unique()

fig2, ax2 = plt.subplots() # Create a new figure and axis for the 2D plot

for index, cluster in enumerate(clusters):
    channel1 = (df[df.iloc[:, 2] == cluster].iloc[:, 0].to_numpy()) # find channel1 value for given cluster
    channel2 = (df[df.iloc[:, 2] == cluster].iloc[:, 1].to_numpy()) # find channel2 value for given cluster
    ax2.scatter(channel1, channel2, color=colors[index]) # applies unique color to cluster

# set axis labels and title
ax2.set_xlabel('Channel 1')
ax2.set_ylabel('Channel 2')
ax2.set_title('2D Clustering')



# 3D Data
# read in the CSV data
df = pd.read_csv("../../data/3d_assay_4.csv")

# get the unique clusters and set the color scheme
clusters = df.iloc[:, -1].unique()

fig3 = plt.figure()  # Create a new figure for the 3D plot
ax3 = fig3.add_subplot(111, projection='3d')

# iterate over the clusters and add each cluster's data to the same axis object
for index, cluster in enumerate(clusters):
    channel1 = df.loc[df.iloc[:, -1] == cluster, df.columns[0]]
    channel2 = df.loc[df.iloc[:, -1] == cluster, df.columns[1]]
    channel3 = df.loc[df.iloc[:, -1] == cluster, df.columns[2]]
    ax3.scatter(channel1, channel2, channel3, c=colors[index])

# set labels for the axes
ax3.set_xlabel('Channel 1')
ax3.set_ylabel('Channel 2')
ax3.set_zlabel('Channel 3')
ax3.set_title('3D Clustering')

# display the plots
plt.show()
