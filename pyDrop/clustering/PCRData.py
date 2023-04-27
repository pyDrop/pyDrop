"""
=========================================================================================
                    PCRData: Data Handling and Clustering Pipeline
=========================================================================================
PCRData: class wrapper for most of the plotting and clustering functionality in pyDrop
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import cluster
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score

from pyDrop.exceptions import (PCRDataReadingException,
                               FeaturesImproperlySpecified,
                               LabelsImproperlySpecified,
                               TooManyDimensions,
                               SupervisedDataRequired,
                               NumClustersModelValueError,
                               ImproperClusterModel)

class PCRData:
    """Loads Data from file or pandas dataframe. Includes options 
    for column naming which is required if input data does not follow
    the format ch1, ch2, ... cluster_* (case in-sensitive).
    Parameters
    ----------
    data : pandas.dfame | str
        pandas dataframe or file path to csv file containing 
    X_columns : list(str)
        column names that correspond to the features. Overrides default names
    y_columns : str
        column name that corresponds to the labels. Overrides default names
    Usage
    -----
    >>> filepath = "data/3d_assay_0.csv"
    >>> data = PCRData(filepath, sep=",", header=True)
    >>> data.plot()
    PCRData also contains functionality to subsample data if true labels are given for testing purposes:
    >>> data.subsample_clusters({1: 1.0, 2: 0, 3: 0.5})
    """
    def __init__(self, data, X_columns: list(str)=[], y_column: str="", verbose: bool=True, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, str):
            try:
                data = pd.read_csv(data, *args, **kwargs)
            except Exception as e:
                raise PCRDataReadingException(e)
        else:
            raise PCRDataReadingException(f"Data must be of the format str or pandas.DataFrame. Type {type(data)} was given.")
        
        self.verbose = verbose
        
        if X_columns: # explicitly given X columns
            if any([name not in data.columns for name in X_columns]): # if not all of X_columns appear 
                ghost_names = [name for name in X_columns if name not in data.columns]
                raise FeaturesImproperlySpecified(f"The given names do not appear in the data feature columns: {ghost_names}")
            else:
                self.X_cols = X_columns
        else: # if no explicitly given X columns, try to infer based on a standard format ch*
            regexp = re.compile(r'ch*', re.I)
            skipped_cols = []
            for name in data.columns:
                if regexp.search(name):
                    X_columns.append(name)
                else:
                    skipped_cols.append(name)
            if skipped_cols and self.verbose:
                raise UserWarning(f"The following columns were skipped when assigning features automatically {skipped_cols}")
            
            self.X_cols = X_columns

        self.supervised = True
        if y_column:
            if y_column not in data.columns:
                raise LabelsImproperlySpecified(f"The given y column ({y_column}) does not appear in the data.")
            else:
                self.y_col = y_column
        else: # if no explicitly given y column, try to infer based on the format column_*
            regexp = re.compile(r'column_*', re.I)
            for name in data.columns:
                if regexp.search(name) and not self.y_col:
                    self.y_col = name
                elif regexp.search(name) and self.y_col:
                    raise LabelsImproperlySpecified(f"2 y columns found: [{self.y_col}, {name}]. Please use the y_column argument to specify which column to use. ")
            if not self.y_col:
                raise UserWarning("No y column found. Now in supervised mode.")
                self.supervised = False # assign false only if no default column found

        self.col_to_id = dict([(col_name, idx) for idx, col_name in enumerate(self.X_cols)])
        
        self.X = data[self.X_cols].to_numpy()
        if self.supervised:
            self.labels = np.unique(self.y)
            self.label_to_id = dict([(label, idx) for idx, label in enumerate(self.labels)])
            self.y = data[self.y_col].applymap(lambda x: self.label_to_id[x]).to_numpy()
        else:
            self.y = np.array([])
        
        self.num_samples, self.num_features = self.X.shape
        self.predicted_num_clusters = self.num_features * 2 # 2 clusters for each channel
    
    def plot(self):
        """Plots data up to 3 dimensions
        Infers the type of scatter plot to give based on the number of features
        and plots the data accordingly. If the data is supervised and true labels
        are given with the input, clusters are given different colors. Else, all
        points are black. 
        Parameters
        ----------
        None
        Returns
        -------
        fig: matplotlib.pyplot.figure
            A figure representation of the data. 
        """
        # List of unique hexcode colors
        colors = ['#0000FF', '#000000', '#FF0000', '#FFC0CB', '#FFA07A', '#FF7F50',
                '#FF4500', '#FFD700', '#FFFF00', '#00FF00', '#32CD32', '#00FFFF',
                '#008080', '#000080', '#8B00FF', '#FF69B4', '#BA55D3', '#800080',
                '#FF6347', '#CD5C5C', '#A0522D', '#696969']
        
        # Create 1,2 or 3 dimensional scatter plot
        fig, ax = plt.subplots()  # Create a new figure and axis for the 1D plot
        if self.supervised:
            sample_color = [colors[cluster] for cluster in self.y]
        else:
            sample_color = "k"

        if self.num_features == 1:
            ax.scatter(range(self.num_samples), self.X, c=sample_color)  # applies unique color to cluster

            # set axis labels and title
            ax.set_xlabel('Droplet')
            ax.set_ylabel(self.X_cols[0])
            ax.set_title('1D Clustering')

        elif self.num_features == 2:
            ax.scatter(self.X[:,0], self.X[:,1] , c=sample_color)  # applies unique color to cluster

            # set axis labels and title
            ax.set_xlabel('Droplet')
            ax.set_ylabel(self.X_cols[0])
            ax.set_ylabel(self.X_cols[1])
            ax.set_title('2D Clustering')

        elif self.num_features == 3:
            ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2], c=sample_color)  # applies unique color to cluster

            # set axis labels and title
            ax.set_xlabel('Droplet')
            ax.set_ylabel(self.X_cols[0])
            ax.set_ylabel(self.X_cols[1])
            ax.set_ylabel(self.X_cols[2])
            ax.set_title('3D Clustering')

        else:
            raise TooManyDimensions(f"Plotting for data of {self.num_features} dimensions is not supported")

        return fig

    # def prevalidator(self):
    #     return True

    def subsample_clusters(self, probs: dict(int, float)={}):
        """Subsamples the data based on an input set of probabilities for each feature
        Mainly used to test algorithms with decreasing number of sample points per cluster
        or to stress-test workflows. Also used to create train-test splits of the data.
        Parameters
        ----------
        probs: dict(int, float)
            Dictionary of probabilities that correspond to different labels to use in
            the subsampling of the data. For example, a probs dict of {1: 1.0, 2: 0.5} 
            indicates that all of label 1 and half of label 2 should be subsampled. 
        Returns
        -------
        sub_X: numpy.ndarray
            subsampled array of shape (?,num_features) where the number of rows depends
            on the number of subsampled samples.
        sub_y: numpy.ndarray
            subsampled array of same length as sub_X.shape[0] corresponding to the labels
            of the data
        """
        
        if not self.supervised:
            raise SupervisedDataRequired("Data is unsupervised. Must have true value labels to use subsample_clusters method.")

        sub_X = np.array([])
        sub_y = np.array([])

        for label, sample_ratio in probs.items():
            cluster_id = self.label_to_id[label]
            cluster_X = self.data.X[self.data.y == cluster_id]
            cluster_y = self.data.y[self.data.y == cluster_id]
            cluster_n = cluster_y.size

            subsample_ids = np.random.choice(cluster_n, size=cluster_n*sample_ratio)
            sub_X = np.vstack([sub_X, cluster_X[subsample_ids]])
            sub_y = np.vstack([sub_y, cluster_y[subsample_ids]])


        return np.random.shuffle(sub_X), np.random.shuffle(sub_y)

class PCREvaluator:
    """Data container and evaluator for ddPCR data
    The main class container for pyDrop containing the majority of testing and
    post-processing functionality for ddPCR data. Loads data from a pre-instantiated
    PCRData object. 
    Parameters
    ----------
    data: PCRData
        PCRData object with or without true value labels. Note that some of the functionality
        or PCREvaluator depends on true value labels for supervised clustering metrics and
        scores, but post-processing, fitting, and predicting cluster scores is still possible
        without true value labels. 
    Usage
    -----
    >>> filepath = "data/3d_assay_0.csv" # filepath containing true value labels
    >>> data = PCRData(filepath, sep=",", header=True)
    >>> pcr = PCREvaluator(data)
    >>> pcr.set_model(sklearn.cluster.KMeans(n_clusters=data.predicted_num_clusters))
    >>> cluster_labels = pcr.fit_predict()
    """
    def __init__(self, data: PCRData):
        self.data = data
        self.model = None
    
    def plot_supervised_metrics(self, models, display=False):
        """Generates a plot of rand, homogeneity, completeness, and v-measure scores
        from a list of models. All given models must have a working fit_predict function
        that fits and then predicts labels for the data. Typically sklearn.cluster 
        functions are used, but any model may work. As this is a supervised scoring 
        function, the input data must be supervised with valid true labels.
        Parameters
        ----------
        models: sklearn.cluster | cluster-algorithm
            A list of sklearn or other models to run supervised testing on. Each model
            must have a working fit_predict function. 
        display: bool
            If True, display a plot summary of the various scores for each model
        Returns
        -------
        results_dict: dict(str, dict(str, float))
            A dictionary summary of the various scores for each model. Format is
            dict[model_name] = {"rand": <rand_score>, "homogeneity": <homogeneity_score>, ...}
        """

        if not self.data.supervised:
            raise SupervisedDataRequired("Data is unsupervised. Must have true value labels to use plot_supervised_metrics method.")

        results = []
        results_dict = {}

        #Determing accuracy of each clustering model with
        score_names = ["rand", "homogeneity", "completeness", "v_measure"]
        for name, model in models.items():
            predictions = model.fit_predict(self.data.X)
            # Creating an array of accuracy scores
            scores = [rand_score(self.data.y, predictions), homogeneity_score(self.data.y, predictions),
                    completeness_score(self.data.y, predictions), v_measure_score(self.data.y, predictions)]
            results_dict[name] = dict([(score_name, score_value) for score_name, score_value in zip(score_names, scores)])
            results.append(scores)

        if display:
            # Making results vector the correct shape
            results = np.transpose(np.array(results))

            #Creating Bar Graph
            labels = list(models.keys())

            x = np.arange(len(labels))  # the label locations
            width = 0.2  # the width of the bars

            fig1, ax1 = plt.subplots()
            rects1 = ax1.bar(x - 2 * width, results[0], width, label='Rand')
            rects2 = ax1.bar(x - width, results[1], width, label='Homogeneity')
            rects3 = ax1.bar(x, results[2], width, label='Completeness')
            rects4 = ax1.bar(x + width, results[3], width, label='V-measure')

            # Add some text for column values centered above each column
            for rect in rects1 + rects2 + rects3 + rects4:
                height = rect.get_height()
                ax1.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax1.set_ylabel('Score')
            ax1.set_title('Clustering Supervised Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.legend(loc='lower right', fontsize=10)
            plt.show()
        
        return results_dict

    def predict_num_clusters(self, method="elbow", display=False):
        """Uses statistical approaches to predict the number of clusters in the data
        Can either use the elbow method or sillohette plot method to infer the
        number of clusters using simple KMeans. Ideally, the number of clusters 
        returned should correspond roughly to the anticipated number of clusters for
        ddPCR data which is 2 times the number of features (2 assays per axis)
        Parameters
        ----------
        method: str
            Either "elbow" or "silhouette" which specifies the method to use to predict the
            number of clusters.
        display: bool
            If True, display a visual summary of the appropriate method. Gives an elbow plot
            if method="elbow" and a silhouette plot if method="silhouette"
        Returns
        -------
        Number of Cluster: int
            The anticipated number of clusters from elbow or silhouette method using 
            sklearn's KMeans algorithm for clustering and accuracy calculations. 
        """
        if method == "elbow":
            return self._elbow_method(graph=display)
        elif method == "silhouette":
            return 
        else:
            raise NumClustersModelValueError("method must either by elbow or silhouette")

    def _elbow_method(self, graph=False):
        # Extracting data and true clustering values from dataframe
        features = self.data.X

        # Run KMeans with different number of clusters
        k_values = range(1, 4*self.data.num_features)
        inertias = []

        for k in k_values:
            kmeans = cluster.KMeans(n_clusters=k)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)

        # Calculate the slopes between adjacent points
        slopes = np.diff(inertias) / np.diff(k_values)

        # Find the index of the point with the largest change in slopes
        largest_change_index = np.argmax(np.abs(np.diff(slopes))) + 1

        # Calculate the largest change in slopes
        largest_change = np.abs(slopes[largest_change_index] - slopes[largest_change_index - 1])

        if graph:
            fig, ax = plt.subplots()
            ax.plot(k_values, inertias, 'bo-')
            ax.set_xlabel('K-value')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method')

            # Add text to the plot
            textstr = 'K-value where the largest change in slope occurs: k={}'.format(k_values[largest_change_index])
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            plt.show()

        return k_values[largest_change_index]
    
    def set_model(self, model):
        """Sets the model to use in clustering operations
        Given model must be an instantiated object with all variables already
        given. May be any of the sklearn.cluster algorithms or any clustering
        algorithm with a valid fit and predict method. 
        Parameters
        ----------
        model: sklearn.cluster... | other clustering algorithm
            Model to use in future data clustering and prediction
        Returns
        -------
        None
        """
        if "fit" not in dir(model):
            raise ImproperClusterModel("given model does not contain a fit method")
        elif "predict" not in dir(model):
            raise ImproperClusterModel("given model does not contain a predict method")
        
        self.model = model
        self.model.fit(self.data.X)
    
    def predict(self, X_data=None):
        """Gives the predictions from the model previously set
        Assumes a previous call to PCREvaluator.set_model has already been made
        and calls the .predict method of the specified model.
        Parameters
        ----------
        X_data: numpy.ndarray
            If given, predicts clusters from the given data instead of the data
            used to train the model. Used if new data needs to be classified using
            the trained model. 
        Returns
        -------
        cluster_ids: numpy.ndarray
            The integer cluster ids from 0 to <num_clusters-1>. 
        """
        if X_data == None:
            X_data = self.data.X
        cluster_ids = self.model.predict(X_data)
        return cluster_ids
    
    def predict_interstitial(self, tol=0.6, use_true_labels: bool=False):
        """Returns the Interstitial fraction of points given a trained clustering algorithm
        and optionally the true value labels of the data. The interstitial fraction of points
        is the fraction of points that are not likely to be assigned to a single cluster. 
        Whether a point is or is not likely to be clustered is set by the tol factor.
        Parameters
        ----------
        tol: float, default=0.6
            Specifies the probability limit below which a point is not likely to be assigned to
            a given cluster. All points below this probability are included in the interstitial 
            fraction. 
        use_true_labels: bool, default=False
            Whether or not to use the true labels in calculating the interstitial fraction. Otherwise
            the cluster assignments using the given clustering model set by PCREvaluator.set_model()
            is used. If True, the data must be supervised and specifiy true cluster labels.
        Returns
        -------
        assignments: numpy.ndarray
            A vector of assigned clusters with indices from 0 to <num_clusters-1> including an additional
            labels of -1 that identifies which points lie in the interstitial fraction or middle reign.
        """
        return
