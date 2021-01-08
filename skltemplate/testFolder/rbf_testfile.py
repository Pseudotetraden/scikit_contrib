from skltemplate.mnist import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from rbflayer import RBFLayer, InitCentersKMeans
from tensorflow.keras.layers import Dense,Softmax
from rbflayer import InitCentersKMeans2


def normalize(x):
    norm1 = x/np.amax(x)
    return norm1

def convert_one_hot_to_number(one_hot):
    return np.array([np.where(r==1)[0][0] for r in one_hot])

x_train, y_train, x_test, y_test = mnist.load_mnist()

x_train = x_train[0 : 6_000]
y_train = y_train[0 : 6_000]
x_test = x_test[0 : 1_000]
y_test = y_test[0 : 1_000]


#x_train = normalize(x_train)
#x_test = normalize(x_test)

#y_train = convert_one_hot_to_number(y_train)
#y_test = convert_one_hot_to_number(y_test)


def load_data():
    X = x_train
    y = y_train
    return X, y


if __name__ == "__main__":

    X, y = load_data()


    model = Sequential()
    rbflayer = RBFLayer(20,
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

    model.fit(x_train, y_train,
              #verbose=1,
              batch_size=50,
              epochs=5)

    y_pred = model.predict(x_test)
    
    



"""
tf.executing_eagerly()

"""


    