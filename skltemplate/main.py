from skltemplate.mnist import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
def plotMap(n, m, som_map):
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

def convert_one_hot_to_number(one_hot):
    return np.array([np.where(r==1)[0][0] for r in one_hot])



x_train, y_train, x_test, y_test = mnist.load_mnist()

x_train = x_train[0:60_000]
y_train = y_train[0:60_000]
x_test = x_test[0:10_000]
y_test = y_test[0:10_000]

x_train = normalize(x_train)
x_test = normalize(x_test)


y_train = convert_one_hot_to_number(y_train)
y_test = convert_one_hot_to_number(y_test)

def test_som_with_keras():
    from som_with_keras import SelfOrganizingMap as SOM
    
    som = SOM(8,8, 0.9, 1, 10)
    som.fit(x_train, y_train)
    
    sommap = som.get_map()
    som_labels = som.label_map_by_most_common(x_train, y_train)
    print(som_labels.reshape(8,8))
    plotMap(8,8,sommap)
    print("end")
    predicted_labels = som.predict(x_test)
    con_mat = confusion_matrix(y_test, predicted_labels, labels=[0,1,2,3,4,5,6,7,8,9])
    diff = predicted_labels - y_test
    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
    print(con_mat)

def test_som_without_keras():
    from som_without_keras import SelfOrganizingMap as SOM
    
    som = SOM(8,8, 0.9, 10, 10)
    som.fit(x_train, y_train)
    
    sommap = som.get_map()
    som_labels = som.label_map_by_most_common(x_train, y_train)
    print(som_labels.reshape(8,8))
    plotMap(8,8,sommap)
    print("end")
    predicted_labels = som.predict(x_test)
    con_mat = confusion_matrix(y_test, predicted_labels, labels=[0,1,2,3,4,5,6,7,8,9])
    diff = predicted_labels - y_test
    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
    print(con_mat)
    
    

def test_rbf_without_keras():
    from rbfn_without_keras import RadialBasisFunctionNetwork as RBF
    
    rbf = RBF(k=30, batch_size=1)
    
    a = rbf.fit(x_train, y_train)
    
    pred = rbf.predict(x_test)
    
    print(classification_report(y_test, pred))
    
    con_mat = confusion_matrix(y_test, pred, labels=[0,1,2,3,4,5,6,7,8,9])
    
    print(con_mat)
    
    diff = pred - y_test
    
    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
    
    print("end")



def test_rbf_with_keras():
    from rbfn_with_keras import RadialBasisFunctionNetwork as RBF
    
    rbf = RBF(number_rbf_kernels=20, epochs=1, batch_size=1)
    
    print(rbf.get_params())
    a = rbf.fit(x_train, y_train)
    pred = rbf.predict(x_test)
    
    #y_test = convert_one_hot_to_number(y_test)
    
    print(classification_report(y_test, pred))
    
    con_mat = confusion_matrix(y_test, pred, labels=[0,1,2,3,4,5,6,7,8,9])
    
    print(con_mat)
    
    print("end")


#test_rbf_with_keras()

test_rbf_without_keras()

#test_som_without_keras()

#test_som_with_keras()