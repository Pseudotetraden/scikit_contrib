from skltemplate.mnist import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from somlayer import SOMLayer
from tensorflow.keras.layers import Dense,Softmax
from tensorflow.keras.initializers import RandomUniform, Initializer
import matplotlib.pyplot as plt
#tf.compat.v1.enable_eager_execution()



def plotMap(n, m, som_map):
    som_map = som_map.numpy()
    rows = n
    cols = m
    axes=[]
    fig=plt.figure()
    for i in range(n):
       for k in range(m):
           index = i*n+k
           image = som_map[index].reshape(28,28)
           axes.append( fig.add_subplot(rows, cols, index+1) )
           #subplot_title=("Subplot"+str(index))
           #axes[-1].set_title(subplot_title)
           axes[-1].axis('off')
           plt.imshow(image)
    fig.tight_layout()
    plt.axis('off')
    plt.show()

def normalize(x):
    norm1 = x/np.amax(x)
    return norm1
# 
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




if __name__ == "__main__":

    X, y = load_data()

    model = Sequential()
    #self, m,n, input_vector_size, learning_rate,num_epoch, initializer=RandomUniform(0.0, 1.0),radius_factor=1.1, gaussian_std = 0.08, **kwargs
    somlayer = SOMLayer(8, 8, 784, 0.9, 5,x_train.shape[0], dynamic=True)
    model.add(somlayer)

    model.compile(loss='mean_squared_error', run_eagerly=True, 
                  optimizer="adam", metrics=["accuracy"])
    #model.run_eagerly = False

    model.fit(x_train, np.full((x_train.shape[0],1), 1.),
              batch_size=10,
              epochs=1,
              verbose=1)
    
    plotMap(8,8, somlayer.get_map())
    
    #y_pred = model.predict(x_test)
    
    

"""
tf.executing_eagerly()

"""
