from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf
import numpy as np

debug = False

def print_formatted(varname, value):
    if debug:
        print(f"\n{varname}:")
        print(value)

class SOMLayer(Layer):

    def __init__(self, m,n, input_vector_size, learning_rate,num_epoch, initializer=RandomUniform(0.0, 1.0),radius_factor=1.1, gaussian_std = 0.08, **kwargs):
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
        super(SOMLayer, self).__init__(**kwargs)
        
        
    def _neuron_locations(self, m, n):
        # Iterate over both dimensions to generate all 2D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
        
    def build(self, input_shape):
        self.map = self.add_weight(name='map',
                                       shape=(self.m* self.n, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        self.count = 0
        super(SOMLayer, self).build(input_shape)
        

    def call(self, x):
        self.feedforward(x)
        self.backprop(self.count, 5)
        self.count += 1
        return self.map
        
    
    def feedforward(self,x):
        self.x = x
        print_formatted("x", x)
        
        #self.map = tf.reshape(self.map, (self.number_nodes, self.map.shape[2]))
        print_formatted("map", self.map)
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.x, axis=1)), 2), 2)
        print_formatted("squared_distance", self.squared_distance)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        print_formatted("bmu_indices", self.bmu_indices)
        print_formatted("location_vects", self.location_vects)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])
        print_formatted("bmu_locs", self.bmu_locs)


    def backprop(self,iter,num_epoch):

        # Update the weigths 
        radius = tf.subtract(self.sigma,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(num_epoch, 1),tf.float32))))
        print_formatted("radius", radius)

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))
        print_formatted("alpha", alpha)
        
        
        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)
        
        print_formatted("bmu_distance_squares",self.bmu_distance_squares)
        #radius = 1.
        

        self.distance_impact_factor = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self.gaussian_std)), 2)))
                    
        print_formatted("distance_impact_factor", self.distance_impact_factor)

        self.learning_rate_op = tf.multiply(self.distance_impact_factor, alpha)
        
        print_formatted("learning_rate_op", self.learning_rate_op)
        
        weight_diff = tf.subtract(tf.expand_dims(self.x, 1), tf.expand_dims(self.map, 0))
        print_formatted("weight_diff", weight_diff)
        
        tmp = tf.multiply(weight_diff, tf.expand_dims(self.learning_rate_op, 2))
        print_formatted("tmp", tmp)
                          
        
        weight_update = tf.reduce_sum(tmp,axis=0)
        print_formatted("weight_update", weight_update)
        
        self.map = self.map + weight_update
        print_formatted("map", self.map)
        print(self.count)

# lambda_g = iterations/self.mapRadius  # λ
# decreaseFactor = math.exp((-1*(float(iteration)/lambda_g)))
# currentLearningRate = self.learningRate * decreaseFactor
# currentRadius = self.mapRadius * decreaseFactor
# distanceImpactFactor = math.exp(-1*(distance/(2*currentRadius**2)))  # theta
# y.weight + distanceImpactFactor * currentLearningRate * (inputVector-y.weight)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.number_cluster)
    
    def get_map(self):
        return self.map
