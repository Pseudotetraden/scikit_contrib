from sklearn.cluster import KMeans
from skltemplate.mnist import mnist
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.gaussian_process.kernels import RBF
from keras.models import Sequential
from keras.layers import  Dense
from keras.wrappers.scikit_learn import KerasClassifier


from skltemplate.mnist import mnist
import numpy as np

"""
def normalize(x):
    norm1 = x/np.amax(x)
    return norm1

def convert_one_hot_to_number(one_hot):
    return np.array([np.where(r==1)[0][0] for r in one_hot])

x_train, y_train, x_test, y_test = mnist.load_mnist()

x_train = x_train[0:60000]
y_train = y_train[0:60000]
x_test = x_test[0:10000]
y_test = y_test[0:10000]

#x_train = normalize(x_train)
#x_test = normalize(x_test)

y_train = convert_one_hot_to_number(y_train)
y_test = convert_one_hot_to_number(y_test)



def c_model():
    model = Sequential()
    model.add(Dense(784, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = KerasClassifier(build_fn=c_model, epochs=50, batch_size=32)

model.fit(x_train, y_train)
    
print("end")

"""
batch_size=2

for i in range(100):
    if ((i % batch_size)+1) == batch_size:
        print(i)