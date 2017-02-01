import tensorflow as tf
import numpy as np
import pandas as pd
import sys

is_adjusted = False

if len(sys.argv) > 1:
    if sys.argv[1] == "true":
        is_adjusted = True


np.random.seed(7)
tf.set_random_seed(7)
init_data = pd.read_csv("./kc_house_data.csv")
## get rid of useless ID attribute

init_data = init_data.drop("id", axis=1)
## get rid of date attribute because I don't want to deal with objects

init_data = init_data.drop("date", axis=1)
## create function to shuffle data randomly and split training and test sets


### TRYING drop more, less significant data, see what happens

init_data = init_data.drop("long", axis=1)
init_data = init_data.drop("condition", axis=1)
init_data = init_data.drop("yr_built", axis=1)
init_data = init_data.drop("sqft_lot15", axis=1)
init_data = init_data.drop("sqft_lot", axis=1)
init_data = init_data.drop("yr_renovated", axis=1)
init_data = init_data.drop("lat", axis=1)



def split_data(data, ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# get our sets; test set is 20% of data

train_set, test_set = split_data(init_data, 0.2)
## separate the data and the labels

data = (train_set.drop("price", axis=1)).values
data_labels = (train_set["price"].copy()).values  #/ 100
data_labels = data_labels.reshape([len(data_labels),1])

### Get number of features
num_features = data.shape[1]

n_samples = data.shape[0]

## Recreating structure of graph to pull saved graph
X_init = tf.placeholder(tf.float32, [None, num_features])
Y_init = tf.placeholder(tf.float32, [None, 1])
x_mean = tf.reduce_mean(X_init, 0, True)
y_mean = tf.reduce_mean(Y_init, 0, True)
X_mz = tf.subtract(X_init, x_mean)
Y_mz = tf.subtract(Y_init, y_mean)
n_samples = tf.constant(n_samples, dtype=tf.float32)
x_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(X_mz, x_mean), 2), 0, True), tf.subtract(n_samples, 1.0))
y_variance = tf.div(tf.reduce_sum(tf.pow(tf.subtract(Y_mz, y_mean), 2), 0, True), tf.subtract(n_samples, 1.0))
X = tf.div(X_mz, tf.sqrt(x_variance))
Y = tf.div(Y_mz, tf.sqrt(y_variance))
W = tf.Variable(tf.random_normal([num_features,1]))
b = tf.Variable(tf.random_normal([1]))
pred = tf.add(tf.matmul(X,W), b)
adjusted_pred = tf.add(tf.multiply(pred, tf.sqrt(y_variance)), y_mean)
adjusted_Y = tf.add(tf.multiply(Y, tf.sqrt(y_variance)), y_mean) 
pow_val = tf.pow(tf.subtract(pred, Y),2)
cost = tf.reduce_mean(pow_val)
true_pred = tf.add(tf.matmul(X_init, W), b)
ss_e = tf.reduce_sum(tf.pow(tf.subtract(Y, pred), 2))
ss_t = tf.reduce_sum(tf.pow(tf.subtract(Y, 0), 2))
if is_adjusted:
    ss_e = tf.reduce_sum(tf.pow(tf.subtract(Y_init, adjusted_pred), 2))
    ss_t = tf.reduce_sum(tf.pow(tf.subtract(Y_init, y_mean), 2))
r2 = tf.subtract(1.0, tf.div(ss_e, ss_t))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()


saver = tf.train.import_meta_graph("./multi_linear_model.ckpt.meta")

## load in from saved model
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, tf.train.latest_checkpoint("./"))
    pred_data = sess.run(pred, feed_dict={X_init:data, Y_init:data_labels})
    if is_adjusted:
        pred_data = sess.run(adjusted_pred, feed_dict={X_init:data, Y_init:data_labels})
    std_y_data = sess.run(Y, feed_dict={Y_init:data_labels}) 
    if is_adjusted:
        std_y_data = data_labels
    rmse = np.sqrt(np.mean(np.power(np.subtract(pred_data, std_y_data), 2)))
    print("rmse of pred_data and std_y_data is: {0}".format(rmse))
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title("Y vs Y-hat")
    plt.plot(std_y_data, "go")
    plt.plot(pred_data,"bo")
    print("R^2 value: {0}".format(sess.run(r2,feed_dict={X_init:data, Y_init:data_labels})) )
    print("Trying Test data..")
    test_data = (test_set.drop("price", axis=1)).values
    test_data_labels = (test_set["price"].copy()).values
    test_data_labels = test_data_labels.reshape([len(test_data_labels), 1])
    test_pred = sess.run(pred, feed_dict={X_init:test_data, Y_init:test_data_labels})
    if is_adjusted:
        test_pred = sess.run(adjusted_pred, feed_dict={X_init:test_data, Y_init:test_data_labels})
    if not is_adjusted:
        test_data_labels = sess.run(Y, feed_dict={Y_init:test_data_labels})
    rmse = np.sqrt(np.mean(np.power(np.subtract(test_pred, test_data_labels), 2)))
    print("rmse of test_data and test_data_labels is: {0}".format(rmse))
    plt.figure(2)
    plt.title("Test data Y vs Y-hat")
    plt.plot(test_data_labels, "go")
    plt.plot(test_pred, "bo")
    plt.show()

