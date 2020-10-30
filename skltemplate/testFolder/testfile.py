from sklearn.cluster import KMeans
from skltemplate.mnist import mnist
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.gaussian_process.kernels import RBF

c = np.array([[0., 0., 0., 0., 0.25, 1., 0.5, 0.], [0., 0., 0., 0., 1., 1., 0.5, 0.], [0., 0., 0., 0.25, 0.5, 1., 0.5, 0.], [0., 0., 0., 0., 0.5, 1, 0.5, 0], [0., 0., 0., 0., 1., 1., 1., 0]])

#x = np.array([[0., 0., 0., 0.25, 0.25, 1., 1., 0.], [0., 0., 0., 0.25, 0.5, 0.5, 1., 0.], [0., 0., 0., 0.25, 0.25, 0.5, 1., 0.], [0., 0., 0., 0., 0.25, 0.5, 0.5, 0.]])
x = np.array([[0., 0., 0., 0.25, 0.25, 1., 1., 0.], [0., 0., 0., 0.25, 0.25, 1., 1., 0.], [0., 0., 0., 0.25, 0.25, 1., 1., 0.], [0., 0., 0., 0.25, 0.25, 1., 1., 0.], [0., 0., 0., 0.25, 0.25, 1., 1., 0.]])

s = np.array([200, 1.7, 1.7, 1.7, 1.7])

w = np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])

rbf = np.array([[1., 2., 3.],[2., 3., 4.],[5., 6., 7.]])


result = rbf @ w


max_values = [np.argmax(c_l) for c_l in c]
"""
for c_l in c:
    max_values2.append(np.argmax())
"""
print("end")