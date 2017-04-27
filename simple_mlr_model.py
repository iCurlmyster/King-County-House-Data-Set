import tensorflow as tf
import pandas as pd
import numpy as np
import sys

is_adjusted = False
do_plot = False
is_verbose = False
if len(sys.argv) > 1:
    if "-adjusted" in sys.argv:
        is_adjusted = True
    if "-plot" in sys.argv:
        do_plot = True
    if "-v" in sys.argv:
        is_verbose = True


np.random.seed(7)
tf.set_random_seed(7)

init_data = pd.read_csv("./kc_house_data.csv")
if is_verbose:
    print("Cols: {0}".format(list(init_data)) )
    ## we see that there is no missing data from any of the remaining attributes to deal with
    print(init_data.info())

## get rid of useless ID attribute
init_data = init_data.drop("id", axis=1)
## get rid of date attribute because I don't want to deal with objects
init_data = init_data.drop("date", axis=1)
init_data = init_data.drop("zipcode", axis=1)

### Experimenting with dropping less significant data
## when uncommented R^2 is around 52%

#init_data = init_data.drop("long", axis=1)
#init_data = init_data.drop("condition", axis=1)
#init_data = init_data.drop("yr_built", axis=1)
#init_data = init_data.drop("sqft_lot15", axis=1)
#init_data = init_data.drop("sqft_lot", axis=1)
#init_data = init_data.drop("yr_renovated", axis=1)
#init_data = init_data.drop("lat", axis=1)
#init_data = init_data.drop("floors", axis=1)
#init_data = init_data.drop("waterfront", axis=1)
#init_data = init_data.drop("bedrooms", axis=1)
#init_data = init_data.drop("view", axis=1)
#init_data = init_data.drop("sqft_basement", axis=1)

###

if is_verbose:
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

# Grab the mean and stddev for us to standardize in the model
mean_offset = train_set.mean()
stddev_offset = train_set.std()
mean_xOffset = (mean_offset.drop("price")).values
stddev_xOffset = (stddev_offset.drop("price")).values
mean_yOffset = np.array([mean_offset["price"].copy()])
stddev_yOffset = np.array([stddev_offset["price"].copy()])

mean_xOffset = mean_xOffset.reshape(1, mean_xOffset.shape[0])
stddev_xOffset = stddev_xOffset.reshape(1, stddev_xOffset.shape[0])
mean_yOffset = mean_yOffset.reshape(1, mean_yOffset.shape[0])
stddev_yOffset = stddev_yOffset.reshape(1, stddev_yOffset.shape[0])

if do_plot:
    ## use pandas scatter matrix plot to get a better idea of data
    from pandas.tools.plotting import scatter_matrix    
    ## attributes to look at
    attr = ["price", "sqft_living", "grade", "sqft_above", "sqft_living15"]
    scatter_matrix(init_data[attr], figsize=(20,8) )

## separate the data and the labels
data = (train_set.drop("price", axis=1)).values
## needed to divide the numbers into smaller values so when working
## with the data it doesn't overflow and cause a 'nan'
data_labels = (train_set["price"].copy()).values  #/ 100
data_labels = data_labels.reshape([len(data_labels),1])

## Get number of features
num_features = data.shape[1]

if is_verbose:
        print("number of features using: {0}".format(num_features))

n_samples = data.shape[0]

X_init = tf.placeholder(tf.float32, [None, num_features])
Y_init = tf.placeholder(tf.float32, [None, 1])

##  Grab the mean and stddev values we took from the training set earlier
x_mean = tf.constant(mean_xOffset, dtype=tf.float32)
y_mean = tf.constant(mean_yOffset, dtype=tf.float32)

x_stddev = tf.constant(stddev_xOffset, dtype=tf.float32)
y_stddev = tf.constant(stddev_yOffset, dtype=tf.float32)

## Making the input have a mean of 0 and a stddev of 1
X = tf.div(tf.subtract(X_init, x_mean), x_stddev)
Y = tf.div(tf.subtract(Y_init, y_mean), y_stddev)

W = tf.Variable(tf.random_normal([num_features,1]))
b = tf.Variable(tf.random_normal([1]))

pred = tf.add(tf.matmul(X,W), b)

# The prediction that has been reverted from the standardization I did for training the data.
# This is the tensor object you want to run through the session to get normal predictions.
adjusted_pred = tf.add(tf.multiply(pred, tf.sqrt(y_variance)), y_mean) 

pow_val = tf.pow(tf.subtract(pred, Y),2)
cost = tf.reduce_mean(pow_val) 

ss_e = tf.reduce_sum(tf.pow(tf.subtract(Y, pred), 2))
ss_t = tf.reduce_sum(tf.pow(tf.subtract(Y, 0), 2))
if is_adjusted:
    ss_e = tf.reduce_sum(tf.pow(tf.subtract(Y_init, adjusted_pred), 2))
    ss_t = tf.reduce_sum(tf.pow(tf.subtract(Y_init, y_mean), 2))
r2 = tf.subtract(1.0, tf.div(ss_e, ss_t))

# This is the adjusted R^2 formula not to be confused with term adjusted that I am using with other variables to signify values
# that have been reverted from their standardized forms.
adjusted_r2 = tf.subtract(1.0, tf.div(tf.div(ss_e, (n_samples - 1.0)), tf.div(ss_t, (n_samples - num_features - 1)) ) )

optimizer = tf.train.AdamOptimizer(1e-1).minimize(cost)

init = tf.global_variables_initializer()

if is_verbose:
    print("initialize sessions")
sess = tf.Session()

sess.run(init)

loss_values = []
if is_verbose:
    print("training...")
num_epochs = 1000
for epoch in range(num_epochs): 
    _, c = sess.run([optimizer, cost], feed_dict={X_init:data, Y_init:data_labels})
    loss_values.append(c)
    sys.stdout.write("Epoch: {0}/{1} cost: {2}\r".format(epoch+1, num_epochs, c))
    sys.stdout.flush()

if is_verbose:
    print("Training done!")
    training = sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})
    print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )
else:
    training = sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})
    print("Final cost: {0}".format(training))
## values from training data
pred_data = sess.run(pred, feed_dict={X_init:data, Y_init:data_labels})
if is_adjusted:
    pred_data = sess.run(adjusted_pred, feed_dict={X_init:data, Y_init:data_labels})
std_y_data = sess.run(Y, feed_dict={Y_init:data_labels}) 
if is_adjusted:
    std_y_data = data_labels
## values from testing data
test_data = (test_set.drop("price", axis=1)).values
test_data_labels = (test_set["price"].copy()).values
test_data_labels = test_data_labels.reshape([len(test_data_labels), 1])
test_pred = sess.run(pred, feed_dict={X_init:test_data, Y_init:test_data_labels})
if is_adjusted:
    test_pred = sess.run(adjusted_pred, feed_dict={X_init:test_data, Y_init:test_data_labels})
if not is_adjusted:
    test_data_labels = sess.run(Y, feed_dict={Y_init:test_data_labels})

print("R^2 value: {0}".format(sess.run(r2,feed_dict={X_init:data, Y_init:data_labels})) )
print("Adjusted R^2 value: {0}".format(sess.run(adjusted_r2, feed_dict={X_init:data, Y_init:data_labels})))
rmse = np.sqrt(np.mean(np.power(np.subtract(pred_data, std_y_data), 2) ))
print("RMSE of Training data set: {0}".format(rmse))
rmse_test = np.sqrt(np.mean(np.power(np.subtract(test_pred, test_data_labels), 2) ))
print("RMSE of Testing data set: {0}".format(rmse_test))
print("RMSE difference test_RMSE - train_RMSE: {0}".format(rmse_test - rmse))

if do_plot:
    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.title("Cost values")
    plt.plot(loss_values)
    
    plt.figure(3)
    plt.title("Y vs Y-hat")
    plt.plot(std_y_data, "go")
    plt.plot(pred_data,"bo")
    
    plt.figure(4)
    plt.title("Test data Y vs Y-hat")
    plt.plot(test_data_labels, "go")
    plt.plot(test_pred, "bo")
    
    plt.show()

sess.close()

