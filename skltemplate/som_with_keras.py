from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Layer, InputSpec
from sklearn.base import BaseEstimator, ClassifierMixin
from rbflayer import RBFLayer, InitCentersKMeans



class SelfOrganizingMap(KerasClassifier, ClassifierMixin, BaseEstimator):
    def __init__(self, map_dimension=(6,6), learning_rate=0.2, 
                 build_fn="__som_model", **kwargs):
        super(SelfOrganizingMap, self).__init__(**kwargs)
        self.build_fn = build_fn
        self.map_dimension = map_dimension
        self.learning_rate = learning_rate
        
    def __som_model(self, x_train):
        model = Sequential()
        rbflayer = RBFLayer(100,
                            10,
                            initializer=InitCentersKMeans(x_train), #WIEDER EINKOMMENTIEREN!!!
                            input_shape=(784,),
                            #dynamic=True
                            )
        model.add(rbflayer)
        model.add(Dense(10))
    
        model.compile(loss='mean_squared_error', #run_eagerly=True, 
                      optimizer="adam", metrics=["accuracy"])
        #model.run_eagerly = True
        return model



