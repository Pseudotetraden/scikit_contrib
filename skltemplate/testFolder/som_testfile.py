from skltemplate.mnist import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from somlayer import SOMLayer
from tensorflow.keras.layers import Dense,Softmax
from tensorflow.keras.initializers import RandomUniform, Initializer

#tf.compat.v1.disable_eager_execution()
"""
class InitTest(Initializer):
        
    def __call__(self, shape, dtype=None):
        return [[3.,4.,5.],[60.,60.,60.],[3.,3.,4.],[2.,2.,3.]]
    

somlayer = SOMLayer(2,2, 3, 0.9,initializer=InitTest(), dynamic=True)

#somlayer.build(tf.constant([4,3]))
#x = tf.constant([[1.,2.,3.], [2.,3.,4.], [3.,4.,5.], [4.,5.,6.]])

somlayer.build(tf.constant([3]))
x = tf.constant([[1.,2.,3.],[2.,3.,4.]])

somlayer.feedforward(x)
somlayer.backprop(1,0)

somlayer.feedforward(x)
somlayer.backprop(1,0)

somlayer.feedforward(x)
somlayer.backprop(1,0)

"""
def normalize(x):
    norm1 = x/np.amax(x)
    return norm1

def convert_one_hot_to_number(one_hot):
    return np.array([np.where(r==1)[0][0] for r in one_hot])

x_train, y_train, x_test, y_test = mnist.load_mnist()

x_train = x_train[0 : 60_000]
y_train = y_train[0 : 60_000]
x_test = x_test[0 : 10_000]
y_test = y_test[0 : 10_000]


x_train = normalize(x_train)
x_test = normalize(x_test)

#y_train = convert_one_hot_to_number(y_train)
#y_test = convert_one_hot_to_number(y_test)


def load_data():
    X = x_train
    y = y_train
    return X, y


import matplotlib.pyplot as plt

def show(image, label):
  print("\nBegin show MNIST image \n")

  image = image.reshape(28,28)
  for row in range(0,28):
    for col in range(0,28):
      print("%02X " % int(image[row][col]), end="")
    print("") 

  print("\ndigit = ", label)

  plt.imshow(image, cmap=plt.get_cmap('gray_r'))
  plt.show()  

  print("\nEnd \n")# -*- coding: utf-8 -*-

if __name__ == "__main__":

    X, y = load_data()

    model = Sequential()
    #self, m,n, input_vector_size, learning_rate,num_epoch, initializer=RandomUniform(0.0, 1.0),radius_factor=1.1, gaussian_std = 0.08, **kwargs
    somlayer = SOMLayer(8, 8, 784, 0.9, 5, dynamic=True)
    model.add(somlayer)

    model.compile(loss='mean_squared_error', run_eagerly=True, 
                  optimizer="adam", metrics=["accuracy"])
    #model.run_eagerly = True

    model.fit(x_train, y_train,
              batch_size=50,
              epochs=5,
              verbose=1)
    
    for weight in somlayer.get_map():
        show(weight, "x")

    #y_pred = model.predict(x_test)
    
    



"""
tf.executing_eagerly()

"""
