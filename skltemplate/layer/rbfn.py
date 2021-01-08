from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer
from sklearn.cluster import KMeans 
from keras import backend as K
import math



class InitCentersKMeans(Initializer):

    def __init__(self, X, n_clusters, max_iter=300):
        self.X = X
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        

    def __call__(self, shape, dtype=None):
        #assert shape[1] == self.X.shape[1]
        km = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        self.inertia = km.inertia_
        return km.cluster_centers_
    
    def get_inertia(self, *args, **kwargs):
        return [math.sqrt(self.inertia/self.X.shape[0])]
    

class RBFLayer(Layer):

    def __init__(self, number_cluster, initializer=RandomUniform(0.0, 1.0), **kwargs):
        self.number_cluster = number_cluster
        self.initializer = initializer
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
  
        super(RBFLayer, self).build(input_shape)
        

    def call(self, x):
        #three_d_centroids = 
        three_d_sub = K.transpose(K.expand_dims(self.centroids)-K.transpose(x))
        diff = K.sum(three_d_sub**2, axis=1)
        result = (1/(K.exp((-1*diff)/(self.std_deviation**2))))
        return result
        

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.number_cluster)
