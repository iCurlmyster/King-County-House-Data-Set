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

n_samples = data.shape[0]

#X = tf.placeholder(tf.float64,[None, 18])
#Y = tf.placeholder(tf.float64,[None, 1])

#W = tf.Variable(tf.random_normal([18,1], dtype=tf.float64))
#b = tf.Variable(tf.zeros([1], dtype=tf.float64))

X_init = tf.placeholder(tf.float32, [None, 18])
Y_init = tf.placeholder(tf.float32, [None, 1])


## calculate mean on the column axis for each column. and I am keeping its deminsions
x_mean = tf.reduce_mean(X_init, 0, True)
y_mean = tf.reduce_mean(Y_init, 0, True)

## Making the input have a mean of 0.
## This is elementwise so it will perform on the correct columns for each row.
X_mz = tf.subtract(X_init, x_mean)
Y_mz = tf.subtract(Y_init, y_mean)

n_samples = tf.constant(n_samples, dtype=tf.float32)

## calculate variance. tf.div performs elementwise. also reduce_sum on column axis and keeping deminsions
x_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(X_mz, x_mean), 2), 0, True), tf.subtract(n_samples, 1.0))
y_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(Y_mz, y_mean), 2), 0, True), tf.subtract(n_samples, 1.0))

## Making the input have a variance of 1
X = tf.div(X_mz, tf.sqrt(x_variance))
Y = tf.div(Y_mz, tf.sqrt(y_variance))

W = tf.Variable(tf.random_normal([18,1]))
b = tf.Variable(tf.random_normal([1]))

pred = tf.add(tf.matmul(X,W), b)

adjusted_pred = tf.add(tf.multiply(pred, tf.sqrt(y_variance)), y_mean)

adjusted_Y = tf.add(tf.multiply(Y, tf.sqrt(y_variance)), y_mean) 

ss_e = tf.reduce_sum(tf.pow(tf.subtract(adjusted_Y, adjusted_pred), 2))
ss_t = tf.reduce_sum(tf.pow(tf.subtract(adjusted_Y, y_mean), 2))
r2 = tf.subtract(1.0, tf.div(ss_e, ss_t))

pow_val = tf.pow(tf.subtract(pred, Y),2)

cost = tf.reduce_mean(pow_val)


## adjusted values never would drop in cost. bounced around too much even with really low learning rate
#adjusted_cost = tf.reduce_mean(tf.pow(tf.subtract(adjusted_pred, adjusted_Y), 2) )

## Learning rate was the problem, it needed to be to the 0.00001 degree
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

#adjusted_optimizer = tf.train.GradientDescentOptimizer(0.000005).minimize(adjusted_cost)

init = tf.global_variables_initializer()


print("initialize sessions")
sess = tf.Session()

sess.run(init)

import math
from tqdm import *

loss_values = []

print("training...")
for epoch in tqdm(range(40000)): 
    _, c = sess.run([optimizer, cost], feed_dict={X_init:data, Y_init:data_labels})
    loss_values.append(c)
    if (epoch+1) % 5000 == 0:
        print("Epoch: {0} cost: {1} W: {2} b: {3}".format(epoch, c, sess.run(W), sess.run(b)))


saver = tf.train.Saver()
save_path = saver.save(sess, "./multi_linear_model.ckpt")
print("Model saved in file: {0}".format(save_path))

## load in from saved model
# with tf.Session as sess:
#  saver = tf.train.import_meta_graph("./multi_linear_model.ckpt.meta")
#  saver.restore(sess, "./multi_linear_model.ckpt")


print("Training done!")
training = sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})
print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )


#print("Testing..")
#print("h(35)={0}; y(35)={1}".format(sess.run(pred,feed_dict={X_init:data[35].reshape([1,18])}), data_labels[35].reshape([1,1]) ))


import matplotlib.pyplot as plt

pred_data = sess.run(pred, feed_dict={X_init:data})
plt.figure(1)
plt.title("Y vs Y-hat")
plt.plot(sess.run(Y, feed_dict={Y_init:data_labels}), "go")
plt.plot(pred_data,"bo")

print("R^2 value: {0}".format(sess.run(r2,feed_dict={X_init:data, Y_init:data_labels})) )
plt.figure(2)
plt.title("Cost values")
plt.plot(loss_values)


plt.show()


print("Trying Test data..")
test_data = (test_set.drop("price", axis=1)).values()
test_data_labels = (test_set["price"].copy()).values()
test_pred = sess.run(pred, feed_dict={X_init:test_data})

plt.figure(3)
plt.title("Test data Y vs Y-hat")
plt.plot(sess.run(Y, feed_dict={Y_init:test_data_labels}), "bo")
plt.plot(test_pred, "go")

sess.close()

