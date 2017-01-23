import tensorflow as tf
import pandas as pd
import numpy as np


np.random.seed(7)
tf.set_random_seed(7)

init_data = pd.read_csv("./kc_house_data.csv")

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


# resize data 
#train_set = train_set.assign(sqft_living=(train_set["sqft_living"] / 10000))
#train_set = train_set.assign(sqft_lot=(train_set["sqft_lot"] / 10000))
#train_set = train_set.assign(sqft_above=(train_set["sqft_above"] / 100))
#train_set = train_set.assign(sqft_basement=(train_set["sqft_basement"] / 100))
#train_set = train_set.assign(sqft_living15=(train_set["sqft_living15"] / 100))
#train_set = train_set.assign(sqft_lot15=(train_set["sqft_lot15"] / 10000))


## separate the data and the labels
data = (train_set.drop("price", axis=1)).values
## needed to divide the numbers into smaller values so when working
## with the data it doesn't overflow and cause a 'nan'
data_labels = (train_set["price"].copy()).values  #/ 100
data_labels = data_labels.reshape([len(data_labels),1])
# reshape for tensorflow model
#data_labels = data_labels.reshape((len(data_labels), 1) )

rng = np.random

n_samples = data.shape[0]

#X = tf.placeholder(tf.float64,[None, 18])
#Y = tf.placeholder(tf.float64,[None, 1])

#W = tf.Variable(tf.random_normal([18,1], dtype=tf.float64))
#b = tf.Variable(tf.zeros([1], dtype=tf.float64))

X = tf.placeholder(tf.float32, [None, 18])
Y = tf.placeholder(tf.float32, [None, 1])

## calculate mean
x_mean = tf.reduce_mean(X)
y_mean = tf.reduce_mean(Y)

## Making the input have a mean of 0
X = tf.subtract(X, x_mean)
Y = tf.subtract(Y, y_mean)

n_samples = tf.constant(n_samples, dtype=tf.float32)

## calculate variance
x_variance = tf.reduce_sum(tf.pow(tf.subtract(X, x_mean), 2)) / tf.subtract(n_samples, 1.0)
y_variance = tf.reduce_sum(tf.pow(tf.subtract(Y, y_mean), 2)) / tf.subtract(n_samples, 1.0)

## Making the input have a variance of 1
X = X / tf.sqrt(x_variance)
Y = Y / tf.sqrt(y_variance)

W = tf.Variable(tf.random_normal([18,1]))
b = tf.Variable(tf.random_normal([1]))

pred = tf.add(tf.matmul(X,W), b)

#two = tf.constant(2.0)
#eight = tf.constant(n_samples, dtype=tf.float32)

abs_val = tf.abs(tf.subtract(pred, Y))

#cost = tf.reduce_sum( tf.abs(tf.subtract(pred, Y))) / tf.mul(two, eight)
cost = tf.reduce_mean(abs_val)
#cost = -tf.reduce_sum(tf.log(pred), reduction_indices=1)


## Learning rate was the problem, it needed to be to the 0.00001 degree
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

init = tf.global_variables_initializer()


print("initialize sessions")
sess = tf.Session()

sess.run(init)

import math
from tqdm import *

print("training...")
for epoch in tqdm(range(25000)): 
    sess.run(optimizer, feed_dict={X:data, Y:data_labels})
    if (epoch+1) % 5000 == 0:
        c = sess.run(cost, feed_dict={X:data, Y:data_labels})
        print("Epoch: {0} cost: {1} W: {2} b: {3}".format(epoch, c, sess.run(W), sess.run(b)))


print("Training done!")
training = sess.run(cost, feed_dict={X:data, Y:data_labels})
print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )


print("Testing..")
print("h(35)={0}; y(35)={1}".format(sess.run(pred,feed_dict={X:data[35].reshape([1,18])}), data_labels[35].reshape([1,1]) ))


import matplotlib.pyplot as plt

pred_data = sess.run(pred, feed_dict={X:data})
plt.plot(sess.run(Y, feed_dict={Y:data_labels}), "go")
plt.plot(pred_data,"bo")
plt.show()


#print("Trying Test data..")
#test_data = (test_set.drop("price", axis=1)).values()
#test_data_labels = (test_set["price"].copy()).values()
#test_pred = sess.run(pred, feed_dict={X:test_data})

#plt.plot(sess.run(Y, feed_dict={Y:test_data_labels}), "bo")
#plt.plot(test_pred, "go")

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


