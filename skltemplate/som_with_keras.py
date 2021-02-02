# Author: Thomas Sommer <thomas.sommer@alumni.fh-aachen.de>

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from layer.som import SOMLayer


class SelfOrganizingMap(KerasClassifier, ClassifierMixin, BaseEstimator):
    """Self Organizing Map clustering.

    This model create clusters using a neighborhood function to preserve the 
    topological properties of the input space.

    Parameters
    ----------
    m : integer
        The number of columns of the map.
        
    n : integer
        The number of rows of the map.
        
    learning_rate : float, optional, default 0.2
        Determines the step size at each iteration.
        
    epochs : integer, optional, default 1
        The total number of iterations the nn is training.
        
    batch_size : integer, optional, default 5
        The number input vectors that processed at the same time. Increasing
        this variable will speed up the calculation, but at the cost of 
        accuracy and increased RAM usage.
        
    radius_factor : float, optional, default 1.1
        Influences the initial radius of the neighbourhood function.
        
    build_fn : function, optional, default _som_model
        Specifies the function which creates the Keras model.

    Notes
    -----
    This implementation works with data represented as dense numpy arrays.
    
    The algorithm can be used for finding and visualizing relationships 
    in a high-dimensional input space.

    References
    ----------
    Zell, Andreas
        "Simulation neuronaler Netze." 4., unveränd. Nachdr. München [u.a.] : 
        Oldenbourg, 2003 — ISBN 3-486-24350-0
    """
    
    def __init__(self, m, n, learning_rate = 0.2 ,epochs=1, batch_size=5, radius_factor=1.1, **kwargs):
        self.m = m
        self.n = n
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.radius_factor = radius_factor
        self.build_fn = self._som_model
        super(SelfOrganizingMap, self).__init__(build_fn=self._som_model, **kwargs)
        
        
    def fit(self, x, y, **kwargs):
        """An implementation of a fitting function for a clustering algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : Optional (Not needed for this algorithm)
            array-like, shape (n_samples,)
            The target values. An array of int.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.set_params(x_train=x, m=self.m, n=self.n, input_vector_size=x.shape[1], learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size)
        self.X_ = x
        super(SelfOrganizingMap, self).fit(x, y, **kwargs)
        
    def get_map(self):
        """Grants Access to the current state of the map
            
        Returns
        -------
        self.som_layer.map: numpy-array
            Returns current state of the map as numpy-array
        """
        return self.som_layer.map.numpy()
    
        
    def _som_model(self, x_train, m, n, input_vector_size, learning_rate, epochs):
        model = Sequential()
        somlayer = SOMLayer(m, n, input_vector_size, learning_rate, x_train.shape[0], dynamic=True)
        model.add(somlayer)
        model.compile(loss='mean_squared_error', run_eagerly=True, 
                      optimizer="adam", metrics=["accuracy"])
        self.som_layer = somlayer
        return model
    
    def label_map_by_most_common(self, x, y):
        """Allows to label the clustermap to use the the prediction method
            
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The labeling input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        
        Returns
        -------
        self.som_layer.map: numpy-array
            Returns the labels of the map as numpy-array
        """
        self.map_labeled_ = True
        return self.som_layer.label_map_by_most_common(x,y)
        
    def predict(self, X):
        """ An implementation of a prediction for a clustering algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """ 
        check_is_fitted(self)
        return self.som_layer.predict(X)

        
        
        
    



