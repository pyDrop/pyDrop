Summary
========

Summary of the project

Section 1 - Data Exploration
============================

Something goes here

Section 1.1 - Visualization
---------------------------
This is an introduction

Section 1.1.1 Something else
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data Visualization

Section 2 - Pre-Processing
==========================

Section 3 - Clustering
======================

Section 3.1 - Existing Algorithms
---------------------------------

Coarse Graining
---------------
CGCluster:
    coarse-grained clustering wrapper for any cluster method
    purpose: Increases the ability of unsupervised models to capture smaller clusters through
    course-graining. Trains on the coarse-grained data and may therefore be less able
    to capture fine-detail decision boundaries
KMCalico:
    (Iterative clustering assisted by coarse-graining) developed in the following paper: doi: 10.1021/acs.analchem.7b02688.
    Purpose: Iteratively course-grains the data and the updates the starting location
    of the K-means clustering algorithm. This method is fast and still technically
    trains on the original data with new center starting locations.

Section 4 - Statistical Analysis
================================

