from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
#from rbflayer import RBFLayer, InitCentersKMeans
from layer.som import SOMLayer
from sklearn.utils.multiclass import unique_labels


class SelfOrganizingMap(KerasClassifier, ClassifierMixin, BaseEstimator):
    def __init__(self, m, n, learning_rate,epochs, batch_size, radius_factor=1.1, gaussian_std = 0.08, **kwargs):
        self.m = m
        self.n = n
        self.learning_rate = learning_rate # alpha
        self.gaussian_std = gaussian_std
        self.epochs = epochs
        self.batch_size = batch_size
        self.radius_factor = radius_factor
        self.build_fn = self.som_model
        super(SelfOrganizingMap, self).__init__(build_fn=self.som_model, **kwargs)
        
    def fit(self, x, y, **kwargs):
        #unique_classes = unique_labels(y) # calculated value of y
        #number_of_classes = len(unique_classes)
        self.set_params(x_train=x, m=self.m, n=self.n, input_vector_size=x.shape[1], learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size)
        self.X_ = x
        super(SelfOrganizingMap, self).fit(x, y, **kwargs)
        
    def get_map(self):
        return self.som_layer.map.numpy()
    
    def som_model(self, x_train, m, n, input_vector_size, learning_rate, epochs):
        model = Sequential()
        somlayer = SOMLayer(m, n, input_vector_size, learning_rate, epochs, x_train.shape[0], dynamic=True)
        model.add(somlayer)
        model.compile(loss='mean_squared_error', run_eagerly=True, 
                      optimizer="adam", metrics=["accuracy"])
        self.som_layer = somlayer
        return model
    
    def label_map_by_most_common(self, x, y):
        return self.som_layer.label_map_by_most_common(x,y)
        
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

        return self.som_layer.predict(X)
        
        
        
    



