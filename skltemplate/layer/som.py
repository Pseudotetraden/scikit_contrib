from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Zeros
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import tensorflow as tf
import numpy as np
from sklearn.utils.multiclass import unique_labels


class SOMLayer(Layer):

    def __init__(self, m,n, input_vector_size, learning_rate,num_epoch, 
                 total_iteration, initializer=RandomUniform(0.0, 1.0),
                 count = 0,radius_factor=1.1, gaussian_std = 0.08, **kwargs):
        self.m = m
        self.n = n
        self.number_nodes = m*n
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.sigma = max(m,n)*radius_factor
        self.gaussian_std = gaussian_std
        self.num_epoch = num_epoch
        self.total_iteration = total_iteration
        self.count = count
        super(SOMLayer, self).__init__(**kwargs)
        
        
    def _neuron_locations(self, m, n):
        # Iterate over both dimensions to generate all 2D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
                
                
    def build(self, input_shape):
        self.map = self.add_weight(name='map',
                                       shape=(self.m * self.n, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        super(SOMLayer, self).build(input_shape)
        
        
    def call(self, x):
        self.feedforward(x)
        self.backprop(self.count, self.num_epoch)
        self.count += x.shape[0]
        return tf.constant([1.])
        
    
    def feedforward(self,x):
        self.x = x
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(
            tf.expand_dims(self.map, axis=0),
            tf.expand_dims(self.x, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(
            self.location_vects,
            self.bmu_indices), [-1, 2])


    def backprop(self,iter,num_epoch):
        
        lambda_g = self.total_iteration/self.sigma  # λ
        
        decreaseFactor = tf.exp((-1*(float(iter)/lambda_g)))

        # Update the weigths 
        
        self.bmu_distance_squares = tf.cast(tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2), "float32")
        
        
        radius = self.sigma * decreaseFactor + float(1e-6)
        alpha = self.alpha * decreaseFactor
        
        self.distance_impact_factor = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, tf.float32)), tf.multiply(
                tf.square(radius), 2)))
                    
        # restrict updates to radius:
        in_radius = tf.cast(
            tf.math.less_equal(
                tf.math.sqrt(self.bmu_distance_squares),
                radius),
            tf.int32)
        
        self.distance_impact_factor = self.distance_impact_factor * tf.cast(in_radius,tf.float32)

        self.learning_rate_op = tf.multiply(self.distance_impact_factor, alpha)
        
        weight_diff = tf.subtract(tf.expand_dims(self.x, 1), tf.expand_dims(self.map, 0))
        
        weight_update = tf.reduce_sum(
            tf.multiply(
                weight_diff, 
                tf.expand_dims(self.learning_rate_op, 2)),
            axis=0)
        
        weight_update_per_batch = tf.divide(weight_update, self.x.shape[0]) 
        
        self.map = tf.math.add(self.map, weight_update_per_batch)


    def compute_output_shape(self, input_shape):
        return tf.constant(1)
    
    
    def get_map(self):
        return self.map
    
    
    def label_map_by_most_common(self, x, y, batch_size=50):
        self.classes_ = unique_labels(y) # calculated value of y
        j= 0
        map_sum = np.zeros(shape=(self.m*self.n, len(self.classes_)))
        while(j < x.shape[0]):
            current_batch_end = j+batch_size if j+batch_size < x.shape[0] else x.shape[0]
            current_batch = x[j:current_batch_end]
            current_batch2 = tf.constant(current_batch,dtype=tf.float32)
            self.feedforward(current_batch2)
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
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        j= 0
        predicted_labels = np.array([])
        while(j < X.shape[0]):
            current_batch_end = j+batch_size if j+batch_size < X.shape[0] else X.shape[0]
            current_batch = X[j:current_batch_end]
            current_batch2 = tf.constant(current_batch,dtype=tf.float32)
            self.feedforward(current_batch2)
            predicted_labels = np.append(predicted_labels, np.take(self.map_labels, self.bmu_indices))
            j = current_batch_end
        
        return predicted_labels
