from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Layer, InputSpec
from sklearn.base import BaseEstimator, ClassifierMixin



class SelfOrganizingMap(KerasClassifier, ClassifierMixin, BaseEstimator):
    def __init__(self, map_dimension=(6,6), learning_rate=0.2, 
                 build_fn="__som_model", **kwargs):
        super(SelfOrganizingMap, self).__init__(**kwargs)
        self.build_fn = build_fn
        self.map_dimension = map_dimension
        self.learning_rate = learning_rate
        
    def __som_model(self):
        model = Sequential()
        model.add(SOMLayer(self.number_of_classes, 0.5))
        model.add(Dense(self.number_of_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model



class SOMLayer(Layer):

    def __init__(self, map_dimension=(6,6), prototypes=None, **kwargs):
        self.map_dimension= map_dimension
        
    def build(self, input_shape):
        """
        set build-flag 
        return void
        build map/prototypes
        """

    def call(self, inputs, **kwargs):
        return "distance_between_inputs_and_prototypes"

    def compute_output_shape(self, input_shape):
        return "output_shape"

    def get_config(self):
        config = {'map_dimension': self.map_dimension}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))