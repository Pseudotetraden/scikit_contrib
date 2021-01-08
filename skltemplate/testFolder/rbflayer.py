from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant, Identity, RandomUniform
from sklearn.cluster import KMeans 
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF
from keras import backend as K
import math

"""
import matplotlib.pyplot as plt

def show(image, label):
  print("\nBegin show MNIST image \n")

  image = image.reshape(28,28)
  for row in range(0,28):
    for col in range(0,28):
      print("%02X " % int(image[row][col]), end="")
    print("") 

  print("\ndigit = ", label)

  plt.imshow(image, cmap=plt.get_cmap('gray_r'))
  plt.show()  

  print("\nEnd \n")# -*- coding: utf-8 -*-

"""

class InitCentersKMeans2():
    def __call__(self, shape, dtype=None):
        return tf.constant([[1.,1.,1.],[1.,1.,1.]])
    
    def get_inertia(self, *args, **kwargs):
        return [1.5]

class InitCentersKMeans(Initializer):

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        

    def __call__(self, shape, dtype=None):
        #assert shape[1] == self.X.shape[1]
        n_centroids = shape[0]
        km = KMeans(n_clusters=n_centroids, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        self.inertia = km.inertia_
        return km.cluster_centers_
    
    def get_inertia(self, *args, **kwargs):
        return [math.sqrt(self.inertia/self.X.shape[0])]
    

class RBFLayer(Layer):

    def __init__(self, number_cluster, number_output_nodes, initializer=RandomUniform(0.0, 1.0), **kwargs):
        self.number_cluster = number_cluster
        self.number_output_nodes = number_output_nodes
        self.initializer = initializer
        self.counter = 0
        super(RBFLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # create cluster center with kmean
        self.centroids = self.add_weight(name='centroids',
                                       shape=(self.number_cluster, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        
        self.std_deviation = self.add_weight(name='std_deviation',
                                       shape=(1),
                                       initializer=self.initializer.get_inertia,
                                       trainable=True)[0]
        """
        self.w = self.add_weight(name='w',
                                     shape=(self.number_cluster,self.number_cluster),
                                     initializer=Identity(), 
                                     #initializer=RandomUniform(0,1),
                                     trainable=True)"""

        super(RBFLayer, self).build(input_shape)
        

    def call(self, x):
        #tf.compat.v1.enable_eager_execution()
        #sess = tf.compat.v1.Session()
        three_d_centroids = K.expand_dims(self.centroids)
        three_d_sub = K.transpose(three_d_centroids-K.transpose(x))
        diff = K.sum(three_d_sub**2, axis=1)
        result = (1/(K.exp((-1*diff)/(self.std_deviation**2))))
        #weighted_result = tf.matmul(result, self.w)
        """
        self.counter += 1
        if self.counter == 1000:
            test = self.centroids.numpy()
            print("end")
        
        print(three_d_centroids.shape)
        print(three_d_sub.shape)
        print(diff.shape)
        print("\n")
        print(result.shape)
        print(weighted_result.shape)
        """
        return result
        



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.number_cluster)
