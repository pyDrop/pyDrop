import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import math
from scipy import stats
from scipy.stats import poisson, norm, chi2, cosine, exponnorm, hypsecant, logistic, genlogistic, dweibull, cauchy, exponweib

# Read the data using pandas and convert to numpy
data = pd.read_csv("../data/3d_assay_2.csv", sep=",", header=0)
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
# Honestly, I had no idea what to do with any of this. I didn't really use cluster labels, I split up the dataframe into a list of lists,
# where the first index position gives the cluster data points

# Functions

def loadLabeledData(file_path, num_clusters, num_channels):
    '''
    Loads labeled data, as formatted by Kari from the file path given. Separates into a list of arrays, each array corresponding to
    a particular cluster

    Input: directory that contains files to be read as a string, number of cluster labels
    Output: two lists, each containing cluster data as numpy arrays, the first indexed as: clusters[cluster number][channel][point] the second
            indexed as clusters[cluster_index][point][channel]
            Both are used in subsequent code to iterate over elements differently
    '''
    # generating a list of cluster names using the BioRad convention
    names = []
    # Loop counts up in binary, int() drops leading zeros
    for k in range(1, num_clusters):
        names.append(str(int('{:04b}'.format(k))))
    num_zeros = 6 - len(names[num_clusters - 2])
    zeros = ''
    for i in range(num_zeros):
        zeros = zeros + '0'
    names = [x + zeros for x in names]
    names = ['0'] + names
    # Generating a list of Channel labels:
    ch_labels = []
    for c in range(num_channels):
        ch_labels.append('Ch' + str(c + 1))
    dframe = pd.read_csv(file_path, header=0)
    clusters = []  # All clusters, points accessed by indexing clusters[cluster index][channel index][point index]
    channels_for_one_cluster = []  # Collects all channels for a single cluster
    for cluster_name in names:
        for channel_name in ch_labels:
            one_channel = dframe[dframe['cluster'] == int(cluster_name)][channel_name].to_numpy()
            channels_for_one_cluster.append(one_channel)
        new_cluster = np.vstack(channels_for_one_cluster)
        channels_for_one_cluster = []
        clusters.append(new_cluster)
    # clusters_other_format is also all clusters, points accessed by indexing clusters[cluster index][channel index][point index]
    clusters_other_format = []
    for cluster in clusters:
        clusters_other_format.append(np.transpose(cluster))
    return clusters, clusters_other_format


def fitCauchy(num_channels, num_clusters, clusters):
    '''
    Input: the number of clusters, the number of channels, and actual cluster dataponts
           Cluster datapoints are in the format clusters[cluster index][channel index][point index]
    Output: A list of fit objects, indexed as fits[cluster index][channel index]
    '''
    # First, calculate summary statistics (mean, std dev) for each cluster, for each channel, storing in an array
    # Cluster stats stores mean and standard deviation, index as clusterStats[cluster #][channel #][mean, std dev]
    clusterStats = np.zeros((num_clusters, num_channels, 2))
    for cluster_index, cluster in enumerate(clusters):
        for channel_index in range(num_channels):
            clusterStats[cluster_index][channel_index][0], clusterStats[cluster_index][channel_index][1] = cluster[
                channel_index].mean(), np.std(cluster[channel_index])
    # For each cluster, fit a distribution to each channel, storing each fit in a list of lists
    fits = []  # indexing is: fits[cluster][channel]
    one_channel_fits = []  # Stores a trio of fits, one for each channel
    # indexing clusters to get correct entry of clusterSumStats
    for cluster_index, cluster in enumerate(clusters):  # outer loop does each cluster
        for channel_index in range(num_channels):  # Inner loop does each channel
            # Compute bounds for given channel
            mean, std = clusterStats[cluster_index][channel_index][0], clusterStats[cluster_index][channel_index][1]
            bounds = [(mean - mean / 2, mean + mean / 2), (std - std / 2, std + std / 2)]
            # Generate fit object for each channel
            fit = stats.fit(cauchy, cluster[channel_index], bounds)
            one_channel_fits.append(fit)
        fits.append(one_channel_fits)
        one_channel_fits = []
    return fits


# Indexing for clusterStats: [cluster][channel][stat]

def fitGeneral_continuous(clusters, num_channels, num_clusters, dist):
    '''
    Same as fitCauchy, but takes an additional parameter, a dist to fit any distribution
    Input: the number of clusters, the number of channels, and actual cluster dataponts
           Cluster datapoints are in the format clusters[cluster index][channel index][point index]
    Output: A list of fit objects, indexed as fits[cluster index][channel index]
    '''
    # First, calculate summary statistics (mean, std dev) for each cluster, for each channel, storing in an array
    # Cluster stats stores mean and standard deviation, index as clusterStats[cluster #][channel #][mean, std dev]
    clusterStats = np.zeros((num_clusters, num_channels, 2))
    for cluster_index, cluster in enumerate(clusters):
        for channel_index in range(num_channels):
            clusterStats[cluster_index][channel_index][0], clusterStats[cluster_index][channel_index][1] = cluster[
                channel_index].mean(), np.std(cluster[channel_index])
    # For each cluster, fit a distribution to each channel, storing each fit in a list of lists
    fits = []  # indexing is: fits[cluster][channel]
    one_channel_fits = []  # Stores a trio of fits, one for each channel
    # indexing clusters to get correct entry of clusterSumStats
    for cluster_index, cluster in enumerate(clusters):  # outer loop does each cluster
        for channel_index in range(num_channels):  # Inner loop does each channel
            # Compute bounds for given channel
            mean, std = clusterStats[cluster_index][channel_index][0], clusterStats[cluster_index][channel_index][1]
            bounds = [(mean - mean / 2, mean + mean / 2), (std - std / 2, std + std / 2)]
            # Generate fit object for each channel
            fit = stats.fit(dist, cluster[channel_index], bounds)
            one_channel_fits.append(fit)
        fits.append(one_channel_fits)
        one_channel_fits = []
    return fits


def oneD_log_likelihood(distribution, values, cont):
    '''
    Input: Frozen distribution, values to fit over as a list or dataframe, boolean to flag if distribution is continuous
    Output: Log likelihood of a given set of values being generated by a distribution
    '''
    ll = 0
    if (cont):
        for i in values:
            ll += distribution.logpdf(i)[0]
    else:
        underflows = []  # If the probability associated with a particular point underflows to zero, the program notes this
        for i in values:
            # Below conditional avoids math domain error and notes the underflow error
            if distribution.pmf(i)[0] == 0:
                # To approximate the underflow point's effect on the likelihood function, approximate the probability as a
                # very small number
                approx_pmf = math.log(2.2250738585072014e-308)  # minimum storable float in python
                underflows.append(i)
                ll += approx_pmf
            else:
                ll += math.log(distribution.pmf(i)[0])
        if len(underflows) > 0:
            # If this message is displayed, the points are vanishingly unlikely to
            print("The probability function of the poisson underflowed", len(underflows), "time(s).")
    return ll


def bestFit(clusters, num_channels, num_clusters, distributions):
    '''
    Input: all datapoints, sorted into a list of clusters, format clusters[cluster_index][channe;_index][point_index] number of channels, number of clusters, distributions to test
    Output: logLikelihoods matrix, indexed as logLikelihoods[distribution][cluster][channel]

    The best distribution is guaged by the sum of the log likelihoods across all data points.
    Note that this function assumes that the only discrete distribution that will be tried is the poisson
    '''
    # indexing for logLikelihoods is logLikelihoods[distribution][cluster][channel]
    logLikelihoods_all = np.zeros([len(distributions), num_clusters, num_channels])
    for dist_index, distribution in enumerate(distributions):
        fits = fitGeneral_continuous(clusters, num_channels, num_clusters, distribution)
        for cluster_index in range(num_clusters):
            for channel_index in range(num_channels):
                logLikelihoods_all[dist_index][cluster_index][channel_index] = oneD_log_likelihood(
                    distribution(fits[cluster_index][channel_index].params), clusters[cluster_index][channel_index],
                    not distribution == poisson)
                # logLikelihoods_all[dist_index][cluster_index][channel_index] = oneD_log_likelihood(fits[cluster_index][channel_index], clusters[cluster_index][channel_index], True)
    # Results (log likelihoods) are returned in array form so they can be indexed by cluster/channel, but are also summed across
    # all clusters/channels so the best distribution for all of the data can be determined.
    # Function
    ll_dict = {}
    for index, dist in enumerate(distributions):
        ll_dict[dist] = np.sum(logLikelihoods_all[index])
        print("The", dist.name, "distribution has log likelihood:", np.sum(logLikelihoods_all[index]))
    return logLikelihoods_all

def IFcalcBounds(fits, clusters, alpha, channel_number, dist):
    '''
    Uses the probability point function and a probabalistic threshold (alpha) to determine which intensity are unlikely given the fitted distribution
        Inputs: Takes vector of fits, vector of clusters, alpha, number of channels, and distribution name
        Outputs: Returns bounds, a numpy array, indexed as bounds[cluster][lower (0) or upper (1)][channel]
    '''
    bounds = np.zeros([num_clusters,2,num_channels])
    for cluster_index, cluster in enumerate(clusters):
        for channel_index in range(channel_number):
            # Populates lower bound and upper bound elements of bounds array, respectivly
            bounds[cluster_index][0][channel_index] = dist.ppf(alpha, (fits[cluster_index][channel_index].params))[0]
            bounds[cluster_index][1][channel_index] = dist.ppf(1-alpha, (fits[cluster_index][channel_index].params))[0]
    return bounds

def IFcalcStrict(bounds, clusters, num_channels):
    '''
    Extracts interstitial points simply and strictly, based on whether or not points are in the bounds returned by IFcalcBounds_numpy
        Inputs: Takes parameters bounds (as returned by function IF calcBounds numpy), cluster, and number of channels
                Note that clusters must be indexed as clusters[cluster index][point index][channel] to compare all channels at once
        Outputs: returns a list of points (as an array) and the cluster index in the form: [array(ch1,...,chN),clusterIndex]
    '''
    # IF_points stored as [ch1,...,chN, clusterIndex]
    IF_points = [] # all points, list format
    # Iterates through each cluster
    for cluster_index, cluster in enumerate(clusters):
        #iterates through each point in each cluster
        for point_index in range(len(cluster)):
        # cluster[index] returns a 3D point
        # Check if point is out of range in any dimension by comparing values in all channels of a given point to the upper and lower bounds
        # for all channels. If all elements are true, point is in range
            lower_logical = [cluster[point_index]>=bounds[cluster_index][0]]
            upper_logical = [cluster[point_index]<=bounds[cluster_index][1]]
        # If one of the elements in the concatenated array is false, np.all will evaluate to false, in which case the point is interstitial
            if(not np.all(np.concatenate((lower_logical, upper_logical)))):
                IF_points.append([cluster[point_index], cluster_index])
    return IF_points

# Plotting code
def IFplot3D(if_points, X,title, label1,label2):
    '''
    Plots 3D scatter plot of data split into two colors
    Input: IF points, as a list of arrays all points X
    Output: none, prints plot
    '''
    # Take the interstitial points, and extract coordinates as a list of arrays
    x, y, z = [x[0] for x in if_points], [x[1] for x in if_points], [x[2] for x in if_points]
    points = np.transpose(np.vstack((x,y,z)))
    X = [i for i in X if i not in points]
    x1, y1, z1 = [x[0] for x in X], [x[1] for x in X], [x[2] for x in X]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Creating plot
    ax.scatter3D(x, y, z, color = "magenta", label=label1)
    ax.scatter3D(x1,y1,z1, color = "blue", label=label2)
    # show plot
    ax.set_xlabel("Channel 1")
    ax.set_ylabel("Channel 2")
    ax.set_zlabel("Channel 3")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()
    return

def IFpoints_analyzeProbability(num_clusters, num_channels, point, dist, fits):
    '''
    Using the fitted distributions, analyzes the probability of each point belonging to each cluster's ditribution, for each
    channel.
    Function gives must likely cluster for the point.
    Input: number of clusters, number of channels, point as output by IFcalcStrict2 (including cluster label), distribution, vector of fits
    Output: bool, True if point should be reclassified, updated cluster index
    '''
    #probabilities indexed as probs[cluster_index][channel_index]
    probs = np.zeros([num_clusters, num_channels])
    for cluster_index in range(num_clusters):
        for channel_index in range(num_channels):
            probs[cluster_index][channel_index]=dist(fits[cluster_index][channel_index].params).logpdf(point[0][channel_index])[0]
    #summing liklihoods of each channel for all the clusters
    collapsed_channels = list(np.sum(probs, axis=1))
    #print(point[1], collapsed_channels.index(max(collapsed_channels)))
    return not point[1]==collapsed_channels.index(max(collapsed_channels)), point[1], collapsed_channels.index(max(collapsed_channels)),point[0]

def IF_analysis(num_clusters, num_channels, if_points,X):
    '''
    Prints
    Input: list of if_points as output by IFpoints_analyzeProbability
    Output: None
    '''
    reclass = [x for x in if_points if x[0]]
    no_reclass = [x for x in if_points if not x[0]]
    interstitialFraction = len(if_points)/X.shape[0]
    print(f"The interstitial fraction of the data is:  {interstitialFraction*100:.2f} %, or",len(if_points),"points are interstitial")
    print(len(reclass), " of the points could be reclassified,", len(no_reclass), " fit best in their current cluster assignments.")
    return


# Function calls
num_clusters = 8
num_channels = 3
clusters, clusters2 = loadLabeledData('../data/3d_assay_2.csv', num_clusters,num_channels)

mat = bestFit(clusters,num_channels,num_clusters,[poisson,cauchy,norm])
#print(np.sum(mat[0]), np.sum(mat[1]), np.sum(mat[2]))
print()

# In terms of the likeihood function, its a tie between poisson and cauchy, but cauchy's percent point function behaves more intuiativly
# so I proceed with that and include a non-generic fitting function that always fits cauchy
fits = fitGeneral_continuous(clusters, num_channels, num_clusters, cauchy)

bounds = IFcalcBounds(fits, clusters2, 0.0005, num_channels, cauchy)
if_points = IFcalcStrict(bounds, clusters2, num_channels)
if_points_sans_label = [x for x,i in if_points]
if_points_reclassed = []
for p in if_points:
    x = IFpoints_analyzeProbability(8,3, p, cauchy, fits)
    if_points_reclassed.append(x)
IF_analysis(num_clusters,num_channels,if_points_reclassed,X)
if_points_reclassed_sans_label = [k for x,i,j,k in if_points_reclassed if x]
print(if_points_reclassed_sans_label)
IFplot3D(if_points_sans_label,X,"Interstitial Points","Iterstitial","In Range")
#IFplot3D(if_points_reclassed_sans_label,X,"Reclassified Points","Candidates for reclassification","Not")
