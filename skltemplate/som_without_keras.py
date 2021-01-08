import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import shuffle
import sys

debug = False
def print_formatted(varname, value):
    if debug:
        shape = value.shape
        print(f"\n{varname}({shape}):")
        print(value)

import matplotlib.pyplot as plt
def plotMap(n, m, som_map):
    rows = n
    cols = m
    axes=[]
    fig=plt.figure()
    for i in range(n):
       for k in range(m):
           index = i*n+k
           image = som_map[index].reshape(28,28)
           axes.append( fig.add_subplot(rows, cols, index+1) )
           #subplot_title=("Subplot"+str(index))
           #axes[-1].set_title(subplot_title)
           axes[-1].axis('off')
           plt.imshow(image)
    fig.tight_layout()
    plt.axis('off')
    plt.show()
    
def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()

class SelfOrganizingMap(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, m,n, learning_rate,num_epoch, batch_size, shuffle=False, radius_factor=1.1, gaussian_std = 0.08, **kwargs):
        self.m = m
        self.n = n
        #self.alpha = learning_rate
        self.learning_rate = learning_rate # alpha
        self.gaussian_std = gaussian_std
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.radius_factor = radius_factor
        self.shuffle = shuffle
        
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
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
        
        #initialize_map
        self.initialize_map()
        self.create_location_vectors()
        self.initialize_sigma()
        
        #run for x epochs
        for i in range(self.num_epoch):
            j= 0
            print(f"Epoch:{i}")
            while(j < self.X_.shape[0]):
                current_batch_end = j+self.batch_size if j+self.batch_size < self.X_.shape[0] else self.X_.shape[0]
                current_batch = self.X_[j:current_batch_end]
                self.feedforward(current_batch)
                self.backprop(j, self.X_.shape[0], current_batch)
                j = current_batch_end 
        # Return the classifier
        return self
    
    def initialize_map(self):
        self.map = np.random.uniform(0.0, 1.0, size=(self.m*self.n, self.X_.shape[1]))
        
    def create_location_vectors(self):
        self.location_vects = np.array(list(self._neuron_locations(self.m, self.n)))
        
    def initialize_sigma(self):
        self.sigma = max(self.m,self.n)*self.radius_factor
    
    def _neuron_locations(self, m, n):
        # Iterate over both dimensions to generate all 2D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
    
   
    def feedforward(self, x):
        self.squared_distance = np.sum(np.power(np.subtract(
            np.expand_dims(self.map, axis=0),
            np.expand_dims(x, axis=1)), 2), 2)
        print_formatted("squared_distance", self.squared_distance)
        self.bmu_indices = np.argmin(self.squared_distance, axis=1)
        print_formatted("bmu_indices", self.bmu_indices)
        self.bmu_locs = np.reshape(np.take(
            self.location_vects,
            self.bmu_indices,
            axis=0), [-1, 2])
        print_formatted("bmu_locs", self.bmu_locs)

    def backprop(self, iter, total_iterations, x):
        
        lambda_g = total_iterations/self.sigma  # λ
        
        decreaseFactor = np.exp((-1*(float(iter)/lambda_g)))

        # Update the weigths 
        self.bmu_distance_squares = np.sum(
                np.power(np.subtract(
                    np.expand_dims(self.location_vects, axis=0),
                    np.expand_dims(self.bmu_locs, axis=1)), 2), 
            2).astype(np.float32)
        
        print_formatted("bmu_distance_squares", self.bmu_distance_squares)
        
        radius = self.sigma * decreaseFactor + float(1e-6)
        alpha = self.learning_rate * decreaseFactor
        
        print_formatted("radius", radius)
        print_formatted("alpha", alpha)
        
        self.distance_impact_factor = np.exp(np.divide(np.negative(
                self.bmu_distance_squares.astype(np.float32)), np.multiply(
                np.square(radius), 2)))
                    
        print_formatted("distance_impact_factor", self.distance_impact_factor)
                    
        # restrict updates to radius:
        in_radius = np.less_equal(
                np.sqrt(self.bmu_distance_squares),
                radius).astype(np.int32)
        
        print_formatted("in_radius", in_radius)
        
        self.distance_impact_factor = self.distance_impact_factor * in_radius.astype(np.float32)
        print_formatted("distance_impact_factor", self.distance_impact_factor)

        self.learning_rate_op = np.multiply(self.distance_impact_factor, alpha)
        print_formatted("learning_rate_op", self.learning_rate_op)
        
        weight_diff = np.subtract(np.expand_dims(x, 1), np.expand_dims(self.map, 0))
        print_formatted("weight_diff", weight_diff)
        
        weight_update = np.sum(
            np.multiply(
                weight_diff, 
                np.expand_dims(self.learning_rate_op, 2)),
            axis=0)
        print_formatted("weight_update", weight_update)
        
        weight_update_per_batch = np.divide(weight_update, x.shape[0]) 
        print_formatted("weight_update_per_batch", weight_update_per_batch)
        
        self.map = np.add(self.map, weight_update_per_batch)
        #plotMap(self.m, self.n, self.map)
        
    def get_map(self):
        return self.map
    
    def label_map_by_most_common(self, x, y):
        j= 0
        map_sum = np.zeros(shape=(self.m*self.n, len(self.classes_)))
        while(j < self.X_.shape[0]):
            current_batch_end = j+self.batch_size if j+self.batch_size < self.X_.shape[0] else self.X_.shape[0]
            current_batch = self.X_[j:current_batch_end]
            self.feedforward(current_batch)
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
        """ A reference implementation of a prediction for a classifier.

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
            self.feedforward(current_batch)
            predicted_labels = np.append(predicted_labels, np.take(self.map_labels, self.bmu_indices))
            j = current_batch_end
        
        return predicted_labels
    
    def __initialize_network(self):
        """
        create map and initialize weights
        """
        pass
    
    def __get_bmu(self, input_vector):
        return "bmu"
    
