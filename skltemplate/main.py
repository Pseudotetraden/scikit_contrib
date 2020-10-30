from skltemplate.mnist import mnist
import numpy as np

def normalize(x):
    norm1 = x/np.amax(x)
    return norm1

def convert_one_hot_to_number(one_hot):
    return np.array([np.where(r==1)[0][0] for r in one_hot])

x_train, y_train, x_test, y_test = mnist.load_mnist()

x_train = x_train[0:6000]
y_train = y_train[0:6000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]

#x_train = normalize(x_train)
#x_test = normalize(x_test)

y_train = convert_one_hot_to_number(y_train)
y_test = convert_one_hot_to_number(y_test)


from rbfn_without_keras import RadialBasisFunctionNetwork as RBF

rbf = RBF(k=11, pseudoinverse=False, std_from_clusters=True, supervised_centroid_calculation=False, batch_size=20)

a = rbf.fit(x_train, y_train, x_test, y_test)

b = rbf.predict(x_test)

diff = b - y_test

print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))

print("end")

"""
#  --------------rbf with keras

from rbfn_with_keras import RadialBasisFunctionNetwork as RBF


rbf = RBF()
a = rbf.fit(x_train, y_train)

print("end")

"""