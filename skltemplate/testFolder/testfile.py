
import tensorflow as tf
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import sys
import time

a = tf.constant([2,3,4])
print(a.numpy())

res = np.array([1,1,1])
bmu_indices = [8, 0 , 1, 8]
labels_map = np.array([[6, 7, 8], [9,10, 11], [12, 13, 14]])

res = np.append(res, np.take(labels_map, bmu_indices))

print(res)

bmu_indices = np.array([1, 3, 0, 2,1,1,1])
bmu_vec = np.identity(8)
bmu = np.take(
            bmu_vec,
            bmu_indices,
            axis=0)

labels = np.array([1, 1, 1, 3,1,8,4])
one_hot_label_vec = np.identity(10)
one_hot_labels = np.take(
            one_hot_label_vec,
            labels,
            axis=0)

#res = np.expand_dims(bmu,axis=2) * one_hot_labels
#res = bmu * np.expand_dims(one_hot_labels, axis=1)
res = np.expand_dims(bmu,axis=2) * np.expand_dims(one_hot_labels, axis=1)

sum_bmu_for_map = np.sum(np.expand_dims(bmu,axis=2) * np.expand_dims(one_hot_labels, axis=1), axis=0)

map_labels = np.argmax(sum_bmu_for_map, axis=1)

print("end")




def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()
    
startProgress("test")
for i in range(100):
    progress(i)
    time.sleep(0.5)
endProgress()


location_vects = [[0, 0], [0, 1], [1, 0], [1, 1]]
bmu_indices = [1,3]

bmu_locs = np.reshape(np.take(
            location_vects,
            bmu_indices,0), [-1, 2])
print(bmu_locs)
bmu_locs = tf.reshape(tf.gather(
            location_vects,
            bmu_indices), [-1, 2])

print(bmu_locs)
a = [1,2,3,4,5,1,2]

b = ["a", "c", "b", "a" ]

c = [[0,0,1], [0,1,0], [0,0,1]]

d = [[0,1],[1,0],[1,1]]

print(unique_labels(c))


print(check_X_y(c, d))



a= tf.Variable([[2.000002, 1., 3.],[3., 3.,4.]])
b = tf.Variable([[1.,2.,3.],[5.,5., 5.],[4.,3.,6.], [2.,3.,4.]])

c = tf.Variable([3.])

#d=a
d = a + float(1e-5)
print(d[0][0])
tf.print(d)

res = tf.math.less_equal(a, c)
print(res)

res2 = tf.cast(res,tf.int32)
print(res2)

a_exp = tf.expand_dims(a,1)
b_exp = tf.expand_dims(b, 0)
print(a_exp)
res = tf.subtract(a_exp, b_exp)
print(res)

learning_rate_op = tf.constant([[1., 0.5, 0.5, 0],[0, 0.5, 0.5, 1.]])

learning_rate_op2 = tf.expand_dims(learning_rate_op, 2)

res2 = tf.multiply(res, learning_rate_op2)
print(res2)
res3 = tf.reduce_sum(res2,axis=0)

print(res3)

