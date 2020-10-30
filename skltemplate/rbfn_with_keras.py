from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Layer
from keras import backend as K
from sklearn.base import BaseEstimator, ClassifierMixin




def create_rbf_model(number_of_classes=10):
    model = Sequential()
    model.add(RBFLayer(number_of_classes, 0.5))
    model.add(Dense(number_of_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

class RadialBasisFunctionNetwork(KerasClassifier): #, ClassifierMixin, BaseEstimator):
    def __init__(self,  number_of_classes=10, learning_rate=0.2, 
                 batch_size=20, build_fn=create_rbf_model, **kwargs):
        #super(RadialBasisFunctionNetwork, self).__init__(**kwargs)
        self.build_fn = build_fn
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size



class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    