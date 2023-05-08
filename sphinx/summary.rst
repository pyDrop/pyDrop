Summary
========

The following python package aims to meet this goal by the design and testing of clustering algorithms and statistical
analysis which improve testing accuracy and reduce computational cost.
Coarse graining is a method by which data sets are binned before clustering is performed. After binning, any existing
algorithm can be used for clustering, although K-means is recommended.

Coarse Graining
---------------
CGCluster:
    Coarse-grained clustering wrapper for any cluster method.
    Purpose: increases the ability of unsupervised models to capture smaller clusters through
    course-graining. Trains on the coarse-grained data and may therefore be less able
    to capture fine-detail decision boundaries.
KMCalico:
    Iterative clustering assisted by coarse-graining. Developed in the following paper: doi: 10.1021/acs.analchem.7b02688.
    Purpose: Iteratively course-grains the data and the updates the starting location
    of the K-means clustering algorithm. This method is fast and still technically
    trains on the original data with new center starting locations.

