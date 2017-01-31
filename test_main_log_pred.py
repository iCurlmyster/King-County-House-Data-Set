import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(7)
tf.set_random_seed(7)
init_data = pd.read_csv("./kc_house_data.csv")
## get rid of useless ID attribute

init_data = init_data.drop("id", axis=1)
## get rid of date attribute because I don't want to deal with objects

init_data = init_data.drop("date", axis=1)
## create function to shuffle data randomly and split training and test sets

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
## needed to divide the numbers into smaller values so when working
## with the data it doesn't overflow and cause a 'nan'

data_labels = (train_set["price"].copy()).values  #/ 100
data_labels = data_labels.reshape([len(data_labels),1])

## load in from saved model
with tf.Session as sess:
    saver = tf.train.import_meta_graph("./multi_linear_model.ckpt.meta")
    saver.restore(sess, "./multi_linear_model.ckpt")
    pred_data = sess.run(true_pred, feed_dict={X_init:data})
    ## TODO still  need to learn how this works and finish implementing


