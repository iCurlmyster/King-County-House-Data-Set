import tensorflow as tf
import pandas as pd
import numpy as np


init_data = pd.read_csv("../DataSets/kc_house_data.csv")

print("Cols: {0}".format(list(init_data)) )

## get rid of useless ID attribute
init_data = init_data.drop("id", axis=1)
## get rid of date attribute because I don't want to deal with objects
init_data = init_data.drop("date", axis=1)

## we see that thtere is no missing data from any of the remaining attributes to deal with
print(init_data.info())


## show the correlation of the attributes against the price attribute
## to see what features are most important to look at.
matrix_corr = init_data.corr()
print(matrix_corr["price"].sort_values(ascending=False))

## create function to shuffle data randomly and split training and test sets
def split_data(data, ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# get our sets; test set is 20% of data
train_set, test_set = split_data(init_data, 0.2)

## use pandas scatter matrix plot to get a better idea of data
## wrote down thoughts in data_processing.txt
#from pandas.tools.plotting import scatter_matrix

## attributes to look at
attr = ["price", "sqft_living", "grade", "sqft_above", "sqft_living15"]
#scatter_matrix(init_data[attr], figsize=(20,8) )

## separate the data and the labels
#data = (train_set.drop("price", axis=1)).values
## needed to divide the numbers into smaller values so when working
## with the data it doesn't overflow and cause a 'nan'
data = (train_set["sqft_living"].copy()).values / 100000
data_labels = (train_set["price"].copy()).values / 1000
# reshape for tensorflow model
#data_labels = data_labels.reshape((len(data_labels), 1) )

rng = np.random

n_samples = data.shape[0]

#X = tf.placeholder(tf.float64,[None, 18])
#Y = tf.placeholder(tf.float64,[None, 1])

#W = tf.Variable(tf.random_normal([18,1], dtype=tf.float64))
#b = tf.Variable(tf.zeros([1], dtype=tf.float64))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(rng.randn())
b = tf.Variable(rng.randn())

pred = tf.add(tf.mul(X,W), b)

two = tf.constant(2.0)
eight = tf.constant(n_samples, dtype=tf.float32)

cost = tf.reduce_sum( tf.pow(tf.subtract(pred, Y), 2)) / tf.mul(two, eight)
#cost = -cost

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()


print("initialize sessions")
sess = tf.Session()

sess.run(init)

import math
from tqdm import *

error = []
d_val = []
dl_val = []

print("training...")
for epoch in range(10000):
    #tmp_d = data[i] #.reshape((1, 18))
    #d_val.append(tmp_d)
    #tmp_dl = data_labels[i] #.reshape((1,1))
    #dl_val.append(tmp_dl)
    _, c = sess.run([optimizer, cost], feed_dict={X:data, Y:data_labels})
    #error.append(c)
    if math.isnan(c):
        print("break")
        break
    if (epoch+1) % 50 == 0:
        c = sess.run(cost, feed_dict={X:data, Y:data_labels})
        print("Epoch: {0:.9f} cost: {1} W: {2} b: {3}\r".format(epoch, c, sess.run(W), sess.run(b)))


print("Training done!")
training = sess.run(cost, feed_dict={X:data, Y:data_labels})
print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )

sess.close()

#with tf.Session() as sess:
#    sess.run(init)
#    for epoch in range(1000):
#        for (x, y) in zip(data, data_labels):
#            sess.run(optimizer, feed_dict={X:x, Y:y})
#        if (epoch+1) % 50 == 0:
#            c = sess.run(cost, feed_dict={X:data, Y:data_labels})
#            print("Epoch: {0:.9f} cost: {1} W: {2} b: {3}".format(epoch, c, sess.run(W), sess.run(b)))
#    
#    
#    print("Training done!")
#    training = sess.run(cost, feed_dict={X:data, Y:data_labels})
#    print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )
#










