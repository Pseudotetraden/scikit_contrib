
import tensorflow as tf
import numpy as np
from test_finished_som import SOM_Layer


a= tf.Variable([[2., 2., 3.],[3., 3.,4.]])
b = tf.Variable([[1.,2.,3.],[5.,5., 5.],[4.,3.,6.], [2.,3.,4.]])

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


somlayer = SOM_Layer(2,2,3,5,0.9,0.5,1.4)
somlayer.feedforward([[1.,2.,3.],[2.,3.,4.]])

def neuron_locations(m, n):
    #""
    #Yields one by one the 2-D locations of the individual neurons in the SOM.
    #"
    # Nested iterations over both dimensions to generate all 2-D locations in the map
    for i in range(m):
        for j in range(n):
            yield np.array([i, j])
                
location_vects = tf.constant(np.array(list(neuron_locations(2, 2))))

print(location_vects)


x = tf.constant([[1,1,1],[1,1,1]])
y = tf.expand_dims(tf.expand_dims(x, 0), 0)


print(y)
