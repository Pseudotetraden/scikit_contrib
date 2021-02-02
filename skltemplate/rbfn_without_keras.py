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


class RadialBasisFunctionNetwork(ClassifierMixin, BaseEstimator):
    """Radial Basis Function Network.

    This Classifier uses the Radial Basis Function as activation Function. The
    output of the network is a linear combination of radial basis functions 
    of the inputs and neuron parameters.

    Parameters
    ----------      
    batch_size : integer, optional, default 5
        The number input vectors that processed at the same time. Increasing
        this variable will speed up the calculation, but at the cost of 
        accuracy and increased RAM usage.
        
    number_rbf_kernels : integer, optional, default 20
        The number specifies how many nodes the hidden layer should have. More
        nodes increases the accuracy of the neuronal network at the cost 
        calculation time.
        
    epochs : integer, optional, default 1
        The total number of iterations the nn is training.
        
    pseudoinverse : boolean, optional, default False
        Allows you to choose between calculating the weights of the nn using a 
        pseudoinverse matrix or by using backpropagation.
        
    learning_rate : float, optional, default 0.1
        Determines the step size at each iteration.
        
    decay_factor: float, optional, default 0.0001
         Determines how fast the learning rate decreases.
         
    activation_function: string, optional, default softmax
        Specifies the activation function used in the output nodes.
        
    supervised_centroid_calculation : boolean, optional, default False
        Group the input data by their classes and get a centroid per class.
        So the number of centroids is set by the number of classes and not by
        the attribute number_rbf_kernels.
        
    shuffle : boolean, optional, default True
        wether or not the order of the input (values and labels) should be 
        randomised. They are randomised in unison.
        
        
    Notes
    -----
    This implementation works with data represented as dense numpy arrays.
    
    The hidden layer is initialized by using kmeans on the whole training set.

    References
    ----------
    Zell, Andreas
        "Simulation neuronaler Netze." 4., unveränd. Nachdr. München [u.a.] : 
        Oldenbourg, 2003 — ISBN 3-486-24350-0
    """
    
    def __init__(self, batch_size=1, epochs=1, number_rbf_kernels=10, pseudoinverse=False,  
                 learning_rate = 0.1, decay_factor=0.0001, activation_function="softmax",
                 supervised_centroid_calculation=False, shuffle=False):
        self.pseudoinverse = pseudoinverse
        self.epochs = epochs
        self.number_rbf_kernels = number_rbf_kernels # number of kernels
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.activation_function = activation_function
        self.supervised_centroid_calculation = supervised_centroid_calculation
        self.shuffle = shuffle
        self.batch_size = batch_size

    def fit(self, X, y):
        """Trains the Radial Basis Function Network.

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
        self._calculate_centroids(X, y)
        
        # Shuffle data
        if self.shuffle:
            X, y = shuffle(X, y)

        
        #dMax = np.max([get_distance(c1, c2) for c1 in self.centroids_ for c2 in self.centroids_])
        #self.std_list = np.repeat(dMax / np.sqrt(2 * self.number_rbf_kernels), self.number_rbf_kernels * len(self.classes_)**int(self.supervised_centroid_calculation))

        self.std_list = np.sqrt(self.inertias_)
            
        rbf_x = self._get_rbf_as_list(X, self.centroids_, self.std_list)
        self._train_weight(rbf_x, y)       
            
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self
    

    def predict(self, X):
        """ Calculates predictions for a given input.

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
        test_weight = self.w
        
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        rbf_test_list = self._get_rbf_as_list(X, self.centroids_, self.std_list)
        
        result_network_input =   rbf_test_list @ self.w
        
        max_values = [np.argmax(res) for res in result_network_input]
        
        #map max_values to classes
        y = []
        for value in max_values:
            y.append(self.classes_[value])
            
        return y

    
    def _calculate_kmean_for_each_label(self, X, y):
        kmeans = KMeans(n_clusters=self.number_rbf_kernels)
        centroids = np.zeros(((len(self.classes_)*self.number_rbf_kernels), X.shape[1]))
        inertias = np.zeros(len(self.classes_)*self.number_rbf_kernels)
        sortOrder = y.argsort()
        X_sorted = X[sortOrder]
        y_sorted = y[sortOrder]
        for i in self.classes_:
            all_occurences = np.where(y_sorted==self.classes_[i])
            x_of_same_label = X_sorted[all_occurences[0][0]:all_occurences[0][-1]]
            kmeans.fit(x_of_same_label)
            centroids[i*self.number_rbf_kernels : i*self.number_rbf_kernels+self.number_rbf_kernels] = kmeans.cluster_centers_
            inertias[i*self.number_rbf_kernels : i*self.number_rbf_kernels+self.number_rbf_kernels] = [kmeans.inertia_/len(x_of_same_label)]*self.number_rbf_kernels
        return centroids, inertias
    
    def _calculate_kmean_for_number_of_labels(self, X, number_of_centroids):
        kmeans = KMeans(n_clusters= number_of_centroids)
        kmeans.fit(X)
        return kmeans.cluster_centers_, np.full(number_of_centroids, kmeans.inertia_/len(X))
        
    
    def _train_weight(self, rbf_x, y):
        if(self.pseudoinverse):
            self._calculate_pseudoinverse(rbf_x, y)
        else:
            self._train_with_backpropagation(rbf_x, y)
            
    def _calculate_pseudoinverse(self, rbf_x, y):            
        self.w = np.linalg.pinv(rbf_x.T @ rbf_x) @ rbf_x.T @ self._convert_to_one_hot(y, len(self.classes_))

    def _train_with_backpropagation(self, rbf_x, y):
                 
        one_hot = self._convert_to_one_hot(y, len(self.classes_))
        
        self.w = np.random.uniform(0,1,(self.number_rbf_kernels*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
        
        avg_weight_update = np.zeros((self.number_rbf_kernels*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
        
        for epoch in range(self.epochs):
            epoch_learning_rate = self.learning_rate
            for i, (current_rbf, one_h) in enumerate(zip(rbf_x, one_hot)):
                
                # forwardpass
                result_network_input =  current_rbf @ self.w 
                result_activation_level = self._get_activation_function(result_network_input)
        
                # backwardpass
                current_rbf = current_rbf.reshape((1,self.number_rbf_kernels*len(self.classes_)**int(self.supervised_centroid_calculation)))
                weight_update = epoch_learning_rate*self._get_delta(self._get_activation_derivate_function(result_network_input), one_h, result_activation_level)* current_rbf.T
                avg_weight_update += weight_update
              
                if ((i % self.batch_size)+1) == self.batch_size:
                    self.w += avg_weight_update
                    avg_weight_update = np.zeros((self.number_rbf_kernels*len(self.classes_)**int(self.supervised_centroid_calculation),len(self.classes_)))
                epoch_learning_rate = epoch_learning_rate / (1+ self.decay_factor)
            
    def _calculate_centroids(self, X, y):
        if self.supervised_centroid_calculation:
            self.centroids_, self.inertias_ = self._calculate_kmean_for_each_label(X, y) # sqrt(intertia) = standard_deviation
        else:
            self.centroids_, self.inertias_ = self._calculate_kmean_for_number_of_labels(X, self.number_rbf_kernels)
    
    def _get_rbf(self, x, c, s, rbf_kernel):
        rbf_kernel.set_params(length_scale= s)
        return rbf_kernel(np.array([c]), x)
    
    def _get_rbf_as_list(self, X, centroids, std_list):
        rbf_kernel = RBF()
        rbf_list = []
        for c, l in zip(centroids, std_list):
            rbf_list.append(self._get_rbf(X, c, l, rbf_kernel)[0])
        rbf_list = np.array(rbf_list)
        rbf_list = rbf_list.T
        return rbf_list
    
    def _convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr
    
    def _get_identity(self, x):
        return x
    
    def _get_identity_derivate(self,x):
        return np.ones(x.size)
    
    def _get_sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _get_sigmoid_derivate(self, x):
        return self._get_sigmoid(x)*(1-self._get_sigmoid(x))
    
    def _get_softmax(self, x):
        return np.exp(x)/(np.sum(np.exp(x)))
    
    def _get_softmax_derivate(self, x):
        return self._get_softmax(x)*(1- self._get_softmax(x))
    
    def _get_activation_function(self, x):
        function_dict = {
            "identity":self._get_identity,
            "sigmoid": self._get_sigmoid,
            "softmax":self._get_softmax
            }
        return function_dict[self.activation_function](x)
        
    
    def _get_activation_derivate_function(self, x):
        function_dict = {
            "identity":self._get_identity_derivate,
            "sigmoid": self._get_sigmoid_derivate,
            "softmax":self._get_softmax_derivate
            }
        return function_dict[self.activation_function](x)
        
    def _get_delta(self, netInput, activationValueShould, activationValueIs):
        t =  netInput * (activationValueShould - activationValueIs)
        return t
    
        