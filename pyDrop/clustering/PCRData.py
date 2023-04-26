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
    >>> 
    """
    def __init__(self, data, X_columns: list(str)=[], y_column: str="", verbose: bool=True, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, str):
            data = pd.read_csv(data, *args, **kwargs)
        else:
            raise Exception
        
        self.verbose = verbose
        
        if X_columns: # explicitly given X columns
            if any([name not in data.columns for name in X_columns]): # if not all of X_columns appear 
                raise Exception
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
                raise UserWarning(skipped_cols)
            
            self.X_cols = X_columns

        self.supervised = True
        if y_column:
            if y_column not in data.columns:
                raise Exception
            else:
                self.y_col = y_column
        else: # if no explicitly given y column, try to infer based on the format column_*
            regexp = re.compile(r'column_*', re.I)
            for name in data.columns:
                if regexp.search(name) and not self.y_col:
                    self.y_col = name
                elif regexp.search(name) and self.y_col:
                    raise Exception
            if not self.y_col:
                raise UserWarning("now in unsupervised mode")
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
            raise UserWarning(f"Plotting for data of {self.num_features} dimensions is not supported")

        return fig

    def prevalidator(self):
        return True

class PCREvaluator:
    def __init__(self, data: PCRData):
        self.data = data
        clusters = self.data.predicted_num_clusters
        self.default_models = [cluster.KMeans(n_clusters=clusters), 
                               cluster.AgglomerativeClustering(n_clusters=clusters),
                               cluster.DBSCAN(), cluster.OPTICS(), 
                               cluster.Birch(n_clusters=clusters)]
    

    def report_supervised_metrics(self, models: list=[]):
        if not self.data.supervised:
            raise Exception
        
        if not models:
            models = self.default_models

        results = []

        #Determing accuracy of each clustering model with
        
        for model in models:
            predictions = model.fit_predict(self.data.X)
            # Creating an array of accuracy scores
            scores = [rand_score(self.data.y, predictions), homogeneity_score(self.data.y, predictions),
                    completeness_score(self.data.y, predictions), v_measure_score(self.data.y, predictions)]
            results.append(scores)

        #Making results vector the correct shape
        results = np.transpose(np.array(results))

        #Creating Bar Graph
        labels = ['KMeans', 'AggClus', 'DBSCAN', 'OPTICS', 'Birch']

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
        
        return fig1

    
