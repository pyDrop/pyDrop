class PCRDataReadingException(Exception):
    """Exception raised when improper data input format is given
    or when file data cannot be successfully read using pandas"""

class FeaturesImproperlySpecified(Exception):
    """Exception raised when not all given X_columns are found
    in the given data. pyDrop makes no inferences on the data columns
    when X_columns are manually specified."""

class LabelsImproperlySpecified(Exception):
    """Exception raised when the given y_columns is not found
    in the given data or when multiple label columns of the format 
    "column_*" are found."""

class TooManyDimensions(Exception):
    """Exception raised when too many dimensions are given in the
    data to effectively visualize in 3 dimensions or less."""

class SupervisedDataRequired(Exception):
    """Exception raised when the called function requires supervised
    ddPCR data with true label assignments but the data is not supervised."""

class NumClustersModelValueError(Exception):
    """Error raised when the incorrect or mistyped model
    is given to the predict the number of clusters"""

class ImproperClusterModel(Exception):
    """Error raised when the given model does not have 
    a valid fit and/or predict method. See sklearn.cluster 
    algorithms for examles of expected model behavior"""

    