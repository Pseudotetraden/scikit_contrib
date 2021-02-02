import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle


class SelfOrganizingMap(ClassifierMixin, BaseEstimator):
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
        
    shuffle : boolean, optional, default True
        wether or not the order of the input (values and labels) should be 
        randomised. They are randomised in unison.
        
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
        
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
        
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

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
    

    def __init__(self, m,n, learning_rate, num_epoch, batch_size, shuffle=False, radius_factor=1.1, **kwargs):
        self.m = m
        self.n = n
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.radius_factor = radius_factor
        self.shuffle = shuffle
        
    def fit(self, X, y=None):
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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if self.shuffle:
            X, y = shuffle(X, y)

        self.X_ = X
        self.y_ = y
        
        self._initialize_map()
        self._create_location_vectors()
        self._initialize_sigma()
        
        for i in range(self.num_epoch):
            j= 0
            print(f"Epoch:{i}")
            while(j < self.X_.shape[0]):
                current_batch_end = j+self.batch_size if j+self.batch_size < self.X_.shape[0] else self.X_.shape[0]
                current_batch = self.X_[j:current_batch_end]
                self._feedforward(current_batch)
                self._backprop(j, self.X_.shape[0], current_batch)
                j = current_batch_end 
        return self
    
    def _initialize_map(self):
        self.map = np.random.uniform(0.0, 1.0, size=(self.m*self.n, self.X_.shape[1]))
        
    def _create_location_vectors(self):
        self.location_vects = np.array(list(self._neuron_locations(self.m, self.n)))
        
    def _initialize_sigma(self):
        self.sigma = max(self.m,self.n)*self.radius_factor
    
    def _neuron_locations(self, m, n):
        # Iterate over both dimensions to generate all 2D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
    
   
    def _feedforward(self, x):
        self.squared_distance = np.sum(np.power(np.subtract(
            np.expand_dims(self.map, axis=0),
            np.expand_dims(x, axis=1)), 2), 2)
        self.bmu_indices = np.argmin(self.squared_distance, axis=1)
        self.bmu_locs = np.reshape(np.take(
            self.location_vects,
            self.bmu_indices,
            axis=0), [-1, 2])

    def _backprop(self, iter, total_iterations, x):
        
        lambda_g = total_iterations/self.sigma  # λ
        
        decreaseFactor = np.exp((-1*(float(iter)/lambda_g)))

        # Update the weigths 
        self.bmu_distance_squares = np.sum(
                np.power(np.subtract(
                    np.expand_dims(self.location_vects, axis=0),
                    np.expand_dims(self.bmu_locs, axis=1)), 2), 
            2).astype(np.float32)
        
        
        radius = self.sigma * decreaseFactor + float(1e-6)
        alpha = self.learning_rate * decreaseFactor
        
        
        self.distance_impact_factor = np.exp(np.divide(np.negative(
                self.bmu_distance_squares.astype(np.float32)), np.multiply(
                np.square(radius), 2)))
                    
                    
        # restrict updates to radius:
        in_radius = np.less_equal(
                np.sqrt(self.bmu_distance_squares),
                radius).astype(np.int32)
        
        
        self.distance_impact_factor = self.distance_impact_factor * in_radius.astype(np.float32)

        self.learning_rate_op = np.multiply(self.distance_impact_factor, alpha)
        
        weight_diff = np.subtract(np.expand_dims(x, 1), np.expand_dims(self.map, 0))
        
        weight_update = np.sum(
            np.multiply(
                weight_diff, 
                np.expand_dims(self.learning_rate_op, 2)),
            axis=0)
        
        weight_update_per_batch = np.divide(weight_update, x.shape[0]) 
        
        self.map = np.add(self.map, weight_update_per_batch)
        #plotMap(self.m, self.n, self.map)
        
    def get_map(self):
        """Grants Access to the current state of the map
            
        Returns
        -------
        map: array-like, shape (n_nodes, n_features)
            Returns current state of the map as numpy-matrix
        """
        return self.map
    
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
        map_labels: numpy-array, shape (n_labels)
            Returns the labels of the map.
        """
        j= 0
        map_sum = np.zeros(shape=(self.m*self.n, len(self.classes_)))
        while(j < self.X_.shape[0]):
            current_batch_end = j+self.batch_size if j+self.batch_size < self.X_.shape[0] else self.X_.shape[0]
            current_batch = self.X_[j:current_batch_end]
            self._feedforward(current_batch)
            bmu_vec = np.identity(self.m*self.n)
            bmu = np.take(
                        bmu_vec,
                        self.bmu_indices,
                        axis=0)
            
            one_hot_label_vec = np.identity(len(self.classes_))
            one_hot_labels = np.take(
                        one_hot_label_vec,
                        y[j:current_batch_end],
                        axis=0)
            map_sum += np.sum(
                        np.expand_dims(bmu,axis=2) 
                        * np.expand_dims(one_hot_labels, axis=1)
                    , axis=0)
            j = current_batch_end
        
        self.map_labels = np.argmax(map_sum, axis=1)
        return self.map_labels
        
        
        
        
    def predict(self, X):
        """ An implementation of a prediction for a cluster algorithm.

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
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        j= 0
        predicted_labels = np.array([])
        while(j < X.shape[0]):
            current_batch_end = j+self.batch_size if j+self.batch_size < X.shape[0] else X.shape[0]
            current_batch = X[j:current_batch_end]
            self._feedforward(current_batch)
            predicted_labels = np.append(predicted_labels, np.take(self.map_labels, self.bmu_indices))
            j = current_batch_end
        
        return predicted_labels
    
    
