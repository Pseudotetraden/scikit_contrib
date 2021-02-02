from keras.layers import Dense, LayerNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from layer.rbfn import RBFLayer, InitCentersKMeans
from sklearn.utils.multiclass import unique_labels


def _create_rbf_model(x_train, number_of_classes, number_rbf_kernels):
    model = Sequential()
    rbflayer = RBFLayer(number_rbf_kernels,
                        initializer=InitCentersKMeans(x_train, n_clusters=number_rbf_kernels), 
                        input_shape=(x_train.shape[1],),
                        )
    model.add(rbflayer)
    model.add(Dense(number_of_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
    return model


class RadialBasisFunctionNetwork(KerasClassifier):
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
        
    build_fn : function, optional, default _create_rbf_model
        Specifies the function which creates the Keras model.
        
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
    def __init__(self, batch_size, number_rbf_kernels=20, build_fn=None, **kwargs):
        self.number_rbf_kernels = number_rbf_kernels
        self.batch_size = batch_size
        self.build_fn = _create_rbf_model
        super(RadialBasisFunctionNetwork, self).__init__(build_fn=_create_rbf_model, **kwargs)
        
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

        Arguments:
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
    
        Returns:
            history : object
                details about the training history at each epoch.
    
        Raises:
            ValueError: In case of invalid shape for `y` argument.
        """
        unique_classes = unique_labels(y)
        number_of_classes = len(unique_classes)
        self.set_params(x_train=x, number_of_classes=number_of_classes, number_rbf_kernels=self.number_rbf_kernels)
        super(RadialBasisFunctionNetwork, self).fit(x, y, **kwargs)
        
        



    
    