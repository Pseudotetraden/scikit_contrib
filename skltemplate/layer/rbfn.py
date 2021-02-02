from keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer
from sklearn.cluster import KMeans 
from keras import backend as K
import math



class InitCentersKMeans(Initializer):
    """Helper Class

    Calculates the centroids for given dataset.

    Parameters
    ----------      
    X : array-like, shape (n_samples, n_features)
        The training input samples.
        
    n_cluster : integer
        Specifies the number of cluster centers.
    
    max_iter : integer, optional, default 300
        Defines the maximum number of iterations for the algorithm to converge.
        
    
    """
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
        """Returns the standard deviation of the centroids.
            
        Returns
        -------
        self : float
            Returns the standard deviation of the centroids.
        """
        return [math.sqrt(self.inertia/self.X.shape[0])]
    

class RBFLayer(Layer):

    """Radial Basis Function Network.

    Forms the core of the Radial Basis Function Network.

    Parameters
    ----------      
        
    number_cluster : integer
        The number specifies how many nodes the hidden layer should have. More
        nodes increases the accuracy of the neuronal network at the cost 
        calculation time.
        
    initalizer : Function, optional, default RandomUniform(0.0, 1.0)
        Provides the initial values for kmeans.
    """
    def __init__(self, number_cluster, initializer=RandomUniform(0.0, 1.0), **kwargs):
        self.number_cluster = number_cluster
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        """Creates the variables of the layer.

        This method is used to create the weights (map) of the SOMLayer and 
        initializes the current_iteration.
    
        Parameters
        ----------
          input_shape: Tensor
              shape of the input matrix/vector
        """
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
        """Contains the layer's logic.

        Parameters
        ----------
        x: Tensor
            Input vectors
            
        **kwargs: Additional keyword arguments.
    
        Returns
        -------
        result : array-like Tensor (n_samples, n_clusters)
            
        """
        three_d_sub = K.transpose(K.expand_dims(self.centroids)-K.transpose(x))
        diff = K.sum(three_d_sub**2, axis=1)
        result = (1/(K.exp((-1*diff)/(self.std_deviation**2))))
        return result
        

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
    
        Returns:
            An input shape tuple.
        """
        return (input_shape[0], self.number_cluster)
