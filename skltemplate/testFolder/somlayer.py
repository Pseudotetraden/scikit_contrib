from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Zeros
import tensorflow as tf
import numpy as np

debug = True

import matplotlib.pyplot as plt
def plotMap(n, m, som_map):
    som_map = som_map.numpy()
    print(som_map[0])
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

def print_formatted(varname, value):
    if debug:
        print(f"\n{varname}:")
        print(value)

class SOMLayer(Layer):

    def __init__(self, m,n, input_vector_size, learning_rate,num_epoch, total_iteration, initializer=RandomUniform(0.0, 1.0),count = 0,radius_factor=1.1, gaussian_std = 0.08, **kwargs):
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
                
    def initialize_zero(self):
        return [0.]
        
    def build(self, input_shape):

        self.map = self.add_weight(name='map',
                                       shape=(self.m * self.n, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        super(SOMLayer, self).build(input_shape)
        

    #build = tf.function(build2)
            
    def call(self, x):
        self.feedforward(x)
        self.backprop(self.count, self.num_epoch)
        self.count += x.shape[0]
        with tf.init_scope():
            a = tf.constant([1.])
        return a
        
    
    def feedforward(self,x):
        self.x = x
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.x, axis=1)), 2), 2)
        print_formatted("squared_distance", self.squared_distance)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        print_formatted("bmu_indices", self.bmu_indices)
        print_formatted("location_vects", self.location_vects)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])
        print_formatted("bmu_locs", self.bmu_locs)

    def backprop(self,iter,num_epoch):
        
        lambda_g = self.total_iteration/self.sigma  # λ
        
        decreaseFactor = tf.exp((-1*(float(iter)/lambda_g)))

        # Update the weigths 
        
        self.bmu_distance_squares = tf.cast(tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2), "float32")
        
        print_formatted("bmu_distance_squares",self.bmu_distance_squares)
        
        radius = self.sigma * decreaseFactor
        alpha = self.alpha * decreaseFactor
        
        #alpha = alpha + float(1e-6)
        radius = radius + float(1e-6)
        """
        print("\n------alpha-------")
        print(alpha)
        print(radius)
        """
        
        self.distance_impact_factor = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, tf.float32)), tf.multiply(
                tf.square(radius), 2)))
                    
        #self.distance_impact_factor = tf.math.add(self.distance_impact_factor, float(1e-6))
                    
        print_formatted("distance_impact_factor", self.distance_impact_factor)
        
        # restrict updates to radius:
        in_radius = tf.cast(
            tf.math.less_equal(
                tf.math.sqrt(self.bmu_distance_squares),
                radius),
            tf.int32)
        
        self.distance_impact_factor = self.distance_impact_factor * tf.cast(in_radius,tf.float32)
        
        print_formatted("distance_impact_factor2", self.distance_impact_factor)

        self.learning_rate_op = tf.multiply(self.distance_impact_factor, alpha)
        #self.learning_rate_op = tf.math.add(self.learning_rate_op, float(1e-6))
        
        learning_rate_op_flat = tf.reshape(self.learning_rate_op, shape=(self.learning_rate_op.shape[0]*self.learning_rate_op.shape[1]))
        
        max_learning_rate = tf.argmax(learning_rate_op_flat)
        min_learning_rate = tf.argmin(learning_rate_op_flat)
        """
        print("\n–––––learning_rate––––––")
        print(learning_rate_op_flat[min_learning_rate])
        print(learning_rate_op_flat[max_learning_rate])
        print("\n")
        """
        print_formatted("learning_rate_op", self.learning_rate_op)
        
        weight_diff = tf.subtract(tf.expand_dims(self.x, 1), tf.expand_dims(self.map, 0))
        print_formatted("weight_diff", weight_diff)
        
        weight_diff_flat = tf.reshape(weight_diff, shape=(weight_diff.shape[0]*weight_diff.shape[1]*weight_diff.shape[2]))
        
        max_update = tf.argmax(weight_diff_flat)
        min_update = tf.argmin(weight_diff_flat)
        """
        print("\n–––––weight_diff–––––––")
        print(weight_diff_flat[min_update])
        print(weight_diff_flat[max_update])
        print("\n")
        """
        weight_update = tf.reduce_sum(
            tf.multiply(
                weight_diff, 
                tf.expand_dims(self.learning_rate_op, 2)),
            axis=0)
        print_formatted("weight_update", weight_update)
        
        weight_update_per_batch = tf.divide(weight_update, self.x.shape[0]) #* 0.5
        
        self.map = tf.math.add(self.map, weight_update_per_batch)
        
        map_flat = tf.reshape(self.map, shape=(self.map.shape[0]*self.map.shape[1]))
        
        min_update = tf.argmin(map_flat)
        max_update = tf.argmax(map_flat)
        """
        print("\n–––––map––––––")
        print(map_flat[min_update])
        print(map_flat[max_update])
        print("\n")
        """
        print_formatted("map", self.map)
        #plotMap(self.n, self.m, self.map)

# lambda_g = iterations/self.mapRadius  # λ
# decreaseFactor = math.exp((-1*(float(iteration)/lambda_g)))
# currentLearningRate = self.learningRate * decreaseFactor
# currentRadius = self.mapRadius * decreaseFactor
# distanceImpactFactor = math.exp(-1*(distance/(2*currentRadius**2)))  # theta
# y.weight + distanceImpactFactor * currentLearningRate * (inputVector-y.weight)

    def compute_output_shape(self, input_shape):
        return tf.constant(1)
    
    def get_map(self):
        return self.map
    
    #  lambda = numer_iterations/radius
