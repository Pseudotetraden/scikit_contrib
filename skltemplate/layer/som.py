from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform
from sklearn.utils.validation import check_array
import tensorflow as tf
import numpy as np
from sklearn.utils.multiclass import unique_labels


class SOMLayer(Layer):
    """Self Organizing Map Layer.

    This layer implements a Self Organizing Map inside a Keras layer.

    Parameters
    ----------
    m : integer
        The number of columns of the map.
        
    n : integer
        The number of rows of the map.
        
    input_vector_size: integer
        The number input nodes
        
    learning_rate : float
        Determines the step size at each iteration.
        
    radius_factor : float, optional, default 1.1
        Influences the initial radius of the neighbourhood function.
        
    total_iteration : integer
        The total number of input vectors (Number of samples).
        
    initializer : function, optional, default RandomUniform(0.0, 1.0)
        Function which creates the inital values for the map.
        
    """
    def __init__(self, m,n, input_vector_size, learning_rate, 
                 total_iteration, initializer=RandomUniform(0.0, 1.0),
                 radius_factor=1.1, **kwargs):
        self.m = m
        self.n = n
        self.number_nodes = m*n
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.sigma = max(m,n)*radius_factor
        self.total_iteration = total_iteration
        super(SOMLayer, self).__init__(**kwargs)
        
        
    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
                
                
    def build(self, input_shape):
        """Creates the variables of the layer.

        This method is used to create the weights (map) of the SOMLayer and 
        initializes the current_iteration.
    
        Parameters
        ----------
          input_shape: Tensor
              shape of the input matrix/vector
        """
        self.current_iteration = 0
        self.map = self.add_weight(name='map',
                                       shape=(self.m * self.n, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        super(SOMLayer, self).build(input_shape)
        
        
    def call(self, x):
        """Contains the layer's logic.

        Parameters
        ----------
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
    
        Returns:
            A Tensor which is unused. Workaround to make Tensorflow available 
            for clustering.
        """
        self._feedforward(x)
        self._backprop(self.current_iteration)
        self.current_iteration += x.shape[0]
        return tf.constant([1.])
        
    
    def _feedforward(self,x):
        self.x = x
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(
            tf.expand_dims(self.map, axis=0),
            tf.expand_dims(self.x, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(
            self.location_vects,
            self.bmu_indices), [-1, 2])


    def _backprop(self,iter):
        
        lambda_g = self.total_iteration/self.sigma
        
        decreaseFactor = tf.exp((-1*(float(iter)/lambda_g)))

        self.bmu_distance_squares = tf.cast(tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2), "float32")
        
        
        radius = self.sigma * decreaseFactor + float(1e-6)
        learning_rate = self.learning_rate * decreaseFactor
        
        self.distance_impact_factor = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, tf.float32)), tf.multiply(
                tf.square(radius), 2)))
                    
        in_radius = tf.cast(
            tf.math.less_equal(
                tf.math.sqrt(self.bmu_distance_squares),
                radius),
            tf.int32)
        
        self.distance_impact_factor = self.distance_impact_factor * tf.cast(in_radius,tf.float32)

        self.learning_rate_op = tf.multiply(self.distance_impact_factor, learning_rate)
        
        weight_diff = tf.subtract(tf.expand_dims(self.x, 1), tf.expand_dims(self.map, 0))
        
        weight_update = tf.reduce_sum(
            tf.multiply(
                weight_diff, 
                tf.expand_dims(self.learning_rate_op, 2)),
            axis=0)
        
        weight_update_per_batch = tf.divide(weight_update, self.x.shape[0]) 
        
        self.map = tf.math.add(self.map, weight_update_per_batch)


    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        If the layer has not been built, this method will call `build` on the
        layer. This assumes that the layer will later be used with inputs that
        match the input shape provided here.
    
        Parameters
        ----------
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.
    
        Returns
        -------
        An input shape value.
        """
        return tf.constant(1)
    
    
    def get_map(self):
        """Grants Access to the current state of the map
            
        Returns
        -------
        map: array-like, shape (n_nodes, n_features)
            Returns current state of the map as Tensor
        """
        return self.map
    
    
    def label_map_by_most_common(self, x, y, batch_size=50):
        """Allows to label the clustermap to use the the prediction method
            
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The labeling input samples.
            
        y : array-like, shape (n_samples,)
            The target values. An array of int.
            
        batch_size : integer, optional, default 50
            Split the data into chunks of this size.
            It is theoretically possible to process the whole dataset at once, 
            but there is a chance that the RAM requirements are not satisfied.
        
        Returns
        -------
        map_labels: numpy-array
            Returns the labels of the map.
        """
        self.classes_ = unique_labels(y) # calculated value of y
        j= 0
        map_sum = np.zeros(shape=(self.m*self.n, len(self.classes_)))
        while(j < x.shape[0]):
            current_batch_end = j+batch_size if j+batch_size < x.shape[0] else x.shape[0]
            current_batch = x[j:current_batch_end]
            current_batch2 = tf.constant(current_batch,dtype=tf.float32)
            self._feedforward(current_batch2)
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
        self.y_ = y
        return self.map_labels
        
        
        
        
    def predict(self, X, batch_size=50):
        """ An implementation of a prediction for a cluster algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
            
        batch_size : integer, optional, default 50
            Split the data into chunks of this size.
            It is theoretically possible to process the whole dataset at once, 
            but there is a chance that the RAM requirements are not satisfied.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """ 

        X = check_array(X)

        j= 0
        predicted_labels = np.array([])
        while(j < X.shape[0]):
            current_batch_end = j+batch_size if j+batch_size < X.shape[0] else X.shape[0]
            current_batch = X[j:current_batch_end]
            current_batch2 = tf.constant(current_batch,dtype=tf.float32)
            self._feedforward(current_batch2)
            predicted_labels = np.append(predicted_labels, np.take(self.map_labels, self.bmu_indices))
            j = current_batch_end
        
        return predicted_labels
