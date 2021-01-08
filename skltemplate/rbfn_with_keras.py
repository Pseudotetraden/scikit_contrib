from keras.layers import Dense, LayerNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from layer.rbfn import RBFLayer, InitCentersKMeans
from sklearn.utils.multiclass import unique_labels


def create_rbf_model(x_train, number_of_classes, number_rbf_kernels):
    print(number_rbf_kernels)
    model = Sequential()
    #model.add(LayerNormalization())
    rbflayer = RBFLayer(number_rbf_kernels,
                        initializer=InitCentersKMeans(x_train, n_clusters=number_rbf_kernels), 
                        input_shape=(x_train.shape[1],),
                       # dynamic=True
                        )
    model.add(rbflayer)
    model.add(Dense(number_of_classes, activation="softmax")) # number_of_classes softmax
    #print(f"number of classes {number_of_classes}")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) #mean_squared_error
    return model


class RadialBasisFunctionNetwork(KerasClassifier): #, ClassifierMixin, BaseEstimator):
    def __init__(self, batch_size, number_rbf_kernels=20, build_fn=None, **kwargs):
        #self.set_params(number_rbf_kernels=number_rbf_kernels, batch_size=batch_size)
        self.number_rbf_kernels = number_rbf_kernels
        #self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.build_fn = create_rbf_model
        super(RadialBasisFunctionNetwork, self).__init__(build_fn=create_rbf_model, **kwargs)
        
    def fit(self, x, y, **kwargs):
        unique_classes = unique_labels(y) # calculated value of y
        number_of_classes = len(unique_classes)
        self.set_params(x_train=x, number_of_classes=number_of_classes, number_rbf_kernels=self.number_rbf_kernels)
        super(RadialBasisFunctionNetwork, self).fit(x, y, **kwargs)
        
        



    
    