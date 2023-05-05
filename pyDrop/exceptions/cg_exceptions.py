class ModelValueError(Exception):
    """Error raised when the incorrect or mistyped model
    is given to the predict method of CGCluster or KMCalico
    """

class AmbiguousCGFunction(Exception):
    """Error raised when coarse graining cannot be done
    either due to improperly specifed binning functions or
    not enough bin functions to accomodate the data.
    """