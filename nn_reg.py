import tensorflow as tf
import pandas as pd
import numpy as np
import sys


np.random.seed(7)
tf.set_random_seed(7)
init_data = pd.read_csv("./kc_house_data.csv")
## get rid of useless ID attribute

init_data = init_data.drop("id", axis=1)
## get rid of date attribute because I don't want to deal with objects

init_data = init_data.drop("date", axis=1)
init_data = init_data.drop("zipcode", axis=1)

def split_data(data, ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# get our sets; test set is 20% of data

train_set, test_set = split_data(init_data, 0.2)
data = (train_set.drop("price", axis=1)).values
## needed to divide the numbers into smaller values so when working
## with the data it doesn't overflow and cause a 'nan'

data_labels = (train_set["price"].copy()).values  #/ 100
data_labels = data_labels.reshape([len(data_labels),1])
## Get number of features

num_features = data.shape[1]
n_samples = data.shape[0]


with tf.device("/gpu:0"):
    X_init = tf.placeholder(tf.float32, [None, num_features])
    Y_init = tf.placeholder(tf.float32, [None, 1])
    
    ## calculate mean on the column axis for each column. and I am keeping its deminsions
    x_mean = tf.reduce_mean(X_init, 0, True)
    y_mean = tf.reduce_mean(Y_init, 0, True)
    
    ## Making the input have a mean of 0.
    ## This is elementwise so it will perform on the correct columns for each row.
    X_mz = tf.subtract(X_init, x_mean)
    Y_mz = tf.subtract(Y_init, y_mean)
    
    # convert to tensor float type
    n_samples = tf.constant(n_samples, dtype=tf.float32)
    
    ## calculate variance. tf.div performs elementwise. also reduce_sum on column axis and keeping deminsions
    x_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(X_mz, x_mean), 2), 0, True), tf.subtract(n_samples, 1.0))
    y_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(Y_mz, y_mean), 2), 0, True), tf.subtract(n_samples, 1.0))
    
    ## Making the input have a variance of 1
    X = tf.div(X_mz, tf.sqrt(x_variance))
    Y = tf.div(Y_mz, tf.sqrt(y_variance))
    
    
    w_1 = tf.Variable(tf.truncated_normal([num_features, 10]))
    b_1 = tf.Variable(tf.truncated_normal([10]))
    
    layer_1 = tf.nn.elu(tf.matmul(X, w_1) + b_1)
    
    w_2 = tf.Variable(tf.truncated_normal([10, 8]))
    b_2 = tf.Variable(tf.truncated_normal([8]))
    
    layer_2 = tf.nn.elu(tf.matmul(layer_1, w_2) + b_2)
    
    w_3 = tf.Variable(tf.truncated_normal([8, 1]))
    b_3 = tf.Variable(tf.truncated_normal([1]))
    
    output_layer = tf.matmul(layer_2, w_3) + b_3
    
    pow_val = tf.pow(tf.subtract(output_layer, Y), 2)
    cost = tf.reduce_mean(pow_val)
    
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost)
    
    init = tf.global_variables_initializer()

loss_values = []
with tf.Session() as sess:
    sess.run(init)
    num_epochs = 7000
    for epoch in range(num_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X_init:data, Y_init:data_labels})
        loss_values.append(c)
        sys.stdout.write("Epoch: {0}/{1} cost: {2}\r".format(epoch+1, num_epochs, c))
        sys.stdout.flush()

    y_values = sess.run(Y, feed_dict={X_init:data, Y_init:data_labels})
    predictions = sess.run(output_layer, feed_dict={X_init:data, Y_init:data_labels})


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(loss_values, "bo")


plt.figure(2)
plt.plot(y_values, "ro")
plt.plot(predictions, "bo")
plt.show()



