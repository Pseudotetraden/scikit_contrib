import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.gaussian_process.kernels import RBF
import math


def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return math.sqrt(sum)

def decrease_function(learning_rate, current_iteration, iterations):
    return learning_rate * math.exp((-1*(float(current_iteration)/iterations)))



class RadialBasisFunctionNetwork(ClassifierMixin, BaseEstimator):

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
    def __init__(self, std_from_cluster=True, pseudoinverse=False, k=10, 
                 learning_rate = 1.0, decay_factor=0.0001, activation_function="softmax",
                 supervised_centroid_calculation=False, shuffle=False, std_from_clusters=False,
                 batch_size=1):
        self.std_from_cluster = std_from_cluster
        self.pseudoinverse = pseudoinverse
        self.k = k # number of kernels
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.activation_function = activation_function
        self.supervised_centroid_calculation = supervised_centroid_calculation
        self.shuffle = shuffle
        self.std_from_clusters = std_from_clusters
        self.batch_size = batch_size

    def fit(self, X, y):
        #self.tX = tX
        #self.ty= ty
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
        
        # Sort X and y based on y    inertia = Variance?
        self.__calculate_centroids(X, y)
        
        # Shuffle data
        if self.shuffle:
            X, y = shuffle(X, y)
        
        #calculate std_deviation
        if not self.std_from_clusters: # Kann eigentlich weg
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids_ for c2 in self.centroids_])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k * len(self.classes_)**int(self.supervised_centroid_calculation))
        else:
            self.std_list = np.sqrt(self.inertias_)
            
        rbf_x = self.__get_rbf_as_list(X, self.centroids_, self.std_list)
        
        self.__train_weight(rbf_x, y)       
            
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self

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
        print(self.w)
        
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        rbf_test_list = self.__get_rbf_as_list(X, self.centroids_, self.std_list)
        
        result_network_input =   rbf_test_list @ self.w
        
        max_values = [np.argmax(res) for res in result_network_input]
        
        #map max_values to classes
        y = []
        for value in max_values:
            y.append(self.classes_[value])
            
        return y

        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        #return self.y_[closest]
    
    def __calculate_kmean_for_each_label(self, X, y):
        kmeans = KMeans(n_clusters=self.k)
        centroids = np.zeros(((len(self.classes_)*self.k), X.shape[1]))
        inertias = np.zeros(len(self.classes_)*self.k)
        sortOrder = y.argsort()
        X_sorted = X[sortOrder]
        y_sorted = y[sortOrder]
        for i in self.classes_:
            all_occurences = np.where(y_sorted==self.classes_[i])
            x_of_same_label = X_sorted[all_occurences[0][0]:all_occurences[0][-1]]
            kmeans.fit(x_of_same_label)
            centroids[i*self.k : i*self.k+self.k] = kmeans.cluster_centers_
            inertias[i*self.k : i*self.k+self.k] = [kmeans.inertia_/len(x_of_same_label)]*self.k
        return centroids, inertias
    
    def __calculate_kmean_for_number_of_labels(self, X, number_of_centroids):
        kmeans = KMeans(n_clusters= number_of_centroids)
        kmeans.fit(X)
        return kmeans.cluster_centers_, np.full(number_of_centroids, kmeans.inertia_/len(X))
        
    
    def __train_weight(self, rbf_x, y):
        if(self.pseudoinverse):
            self.__calculate_pseudoinverse(rbf_x, y)
        else:
            self.__train_with_backpropagation(rbf_x, y)
            
    def __calculate_pseudoinverse(self, rbf_x, y):            
        self.w = np.linalg.pinv(rbf_x.T @ rbf_x) @ rbf_x.T @ self.__convert_to_one_hot(y, len(self.classes_))

    def __train_with_backpropagation(self, rbf_x, y):
                 
        one_hot = self.__convert_to_one_hot(y, len(self.classes_))
        
        self.w = np.random.uniform(0,1,(self.k*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
        
        avg_weight_update = np.zeros((self.k*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
        
        for i, (current_rbf, one_h) in enumerate(zip(rbf_x, one_hot)):
            # forwardpass
            result_network_input =  current_rbf @ self.w 
            result_activation_level = self.get_activation_function(result_network_input)
    
            # backwardpass
            current_rbf = current_rbf.reshape((1,self.k*len(self.classes_)**int(self.supervised_centroid_calculation)))
            weight_update = self.learning_rate*self.get_delta(self.get_activation_derivate_function(result_network_input), one_h, result_activation_level)* current_rbf.T
            avg_weight_update += weight_update
          
            if ((i % self.batch_size)+1) == self.batch_size:
                self.w += avg_weight_update
                avg_weight_update = np.zeros((self.k*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
            self.learning_rate = self.learning_rate / (1+ self.decay_factor)
            
    def __calculate_centroids(self, X, y):
        if self.supervised_centroid_calculation:
            self.centroids_, self.inertias_ = self.__calculate_kmean_for_each_label(X, y) # sqrt(intertia) = standard_deviation
        else:
            self.centroids_, self.inertias_ = self.__calculate_kmean_for_number_of_labels(X, self.k)
    
    def __get_rbf(self, x, c, s, rbf_kernel):
        rbf_kernel.set_params(length_scale= s)
        return rbf_kernel(np.array([c]), x)
    
    def __get_rbf_as_list(self, X, centroids, std_list):
        rbf_kernel = RBF()
        rbf_list = []
        for c, l in zip(centroids, std_list):
            rbf_list.append(self.__get_rbf(X, c, l, rbf_kernel)[0])
        rbf_list = np.array(rbf_list)
        rbf_list = rbf_list.T
        return rbf_list
    
    def __convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr
    
    def get_identity(self, x):
        return x
    
    def get_identity_derivate(self,x):
        return np.ones(x.size)
    
    def get_sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def get_sigmoid_derivate(self, x):
        return self.get_sigmoid(x)*(1-self.get_sigmoid(x))
    
    def get_softmax(self, x):
        return np.exp(x)/(np.sum(np.exp(x)))
    
    def get_softmax_derivate(self, x):
        return self.get_softmax(x)*(1- self.get_softmax(x))
    
    def get_activation_function(self, x):
        function_dict = {
            "identity":self.get_identity,
            "sigmoid": self.get_sigmoid,
            "softmax":self.get_softmax
            }
        return function_dict[self.activation_function](x)
        
    
    def get_activation_derivate_function(self, x):
        function_dict = {
            "identity":self.get_identity_derivate,
            "sigmoid": self.get_sigmoid_derivate,
            "softmax":self.get_softmax_derivate
            }
        return function_dict[self.activation_function](x)
        
    def get_delta(self, netInput, activationValueShould, activationValueIs):
        t =  netInput * (activationValueShould - activationValueIs)
        return t
    
        