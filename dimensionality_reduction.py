#Image Dimensionality Reduction

# Prepare Data
# Design Auto Encoder
# Train Auto Encoder
# Use Encoder level from Auto Encoder
# Use Encoder to obtain reduced dimensionality data for train and test sets

# Commented out IPython magic to ensure Python compatibility.
#import the necessary libraries
import matplotlib
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor

#data preparation
cols = ['#1FC17B', '#78FECF', '#555B6E', '#CC998D', '#429EA6',
        '#153B50', '#8367C7', '#EE6352', '#C287E8', '#F0A6CA',
        '#521945', '#361F27', '#828489', '#9AD2CB', '#EBD494',
        '#53599A', '#80DED9', '#EF2D56', '#446DF6', '#AF929D']

# make_blob gives you back the X values of being the location of this data and y being a class label of this data.
# one of the parameters it make is the number of features which talks about the dimensionality of our data.
# For this purpose, let's choose 50.
# 50 is good amount considering we wil transform this in two dimensional space, fro it to be a good trade off.
# Centers parameters pertains to how many centers of data or how many different clusters
X, y = make_blobs(n_features=50, centers=20, # this translates to our data will have 50 dimensions accross 20 different classes or clusters.
                  n_samples=20000, # you can choose more but it make slightly longer to run. If you choose less, it may not be enough to train a good model.
                  cluster_std=0.2, # the standard deviation of the point between each of the centers.
                  center_box=[-1, 1], # not strictly necessary. We'll just use this to constrain our centers within reasonable location.
                  random_state=17) # for the sake of reproducibility so we can have the same data produced if we run this notebook again.

X[0]
# we get 50 dimensional array

# split the dataset for testing and training
# in cetain ML apps,you might have a validation and a test set, in our case we are just worried about two differnet validations
# considering that we know our dataset is coming from a uniform distribution, we're relatively confident.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17)

# note that each centers could be located somewhere -1 and 1 in a square pattern in each of the 50 dimensions.

# scale the data
scaler = MinMaxScaler() # this will scale each feature. Meaning each dimension will be converted from its existing distribution to be a value between 0 and 1.

# fit the scaler and transform our test set
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# we'll use Regressor function because this is no longer a classification task
# because our inputs are real value and it kinda acts like regression task like
# you want to predict a vector values
autoencoder = MLPRegressor(alpha=1e-15,
                           hidden_layer_sizes=(50, 100, 50, 2, 50, 100, 50),
                           random_state=1, max_iter=20000)
# fit our training data
autoencoder.fit(X_train, X_train) # our input will be train data and our output will also be our train data.

# Dimensionality reduction using encoder of AutoEncoders.
W = autoencoder.coefs_ # W represents weight
biases = autoencoder.intercepts_

for w in W:
    print(w.shape)

encoder_weights = W[0:4]
encoder_biases = biases[0:4]

# create a function where we can loop each of our ecoder weight and encoder biases
def encode(encoder_weights, encoder_biases, data):
    res_ae = data
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index+1 == len(encoder_weights):
            res_ae = res_ae@w+b
        else:
            res_ae = np.maximum(0, res_ae@w+b)

    return res_ae

res_ae = encode(encoder_weights, encoder_biases, X_test) # X-test is our data that our model has not seen before training.

# This is our encoding function. Ths is what will take a 50 dimensional pice of data and convert it ot a list of 2 dimensional point

print("The final reduced dimensionality of the 2000 samples are:")
res_ae.shape