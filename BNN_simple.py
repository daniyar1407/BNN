#!/usr/bin/env python3

import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# hide all the warnings
# not safe!
np.seterr(all='warn')

# deterministic behaviour
np.random.seed(25)

def load_data(load_keras=False):
  if (load_keras):
    # load data using keras
    print("loading data from keras.datasets...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # transform imported dataset
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32')/255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32')/255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
  else:
    # manually load data
    train_data = pd.read_csv("mnist_training.csv")

    for column in train_data:
      train_data[column] = pd.to_numeric(train_data[column], errors="coerce").fillna(0)

    # label manually loaded training data
    images = train_data.iloc[:4000,1:]
    labels = train_data.iloc[:4000,:1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size = 0.2, random_state=25)
  
    # switch from pandas data frames to numpy arrays
    train_images = train_images.values
    test_images = test_images.values
    train_labels = train_labels.values.ravel()
    test_labels = test_labels.values.ravel()

    # categorize inputs
    # train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)

    encoder = LabelEncoder() # integer encoder
    onehot_encoder = OneHotEncoder(sparse=False) # binary encoder

    train_labels = encoder.fit_transform(train_labels)  
    test_labels = encoder.fit_transform(test_labels)
    train_labels = train_labels.reshape(len(train_labels), 1)
    train_labels = onehot_encoder.fit_transform(train_labels)
    test_labels = test_labels.reshape(len(test_labels), 1)
    test_labels = onehot_encoder.fit_transform(test_labels)

  return train_images, test_images, train_labels, test_labels

def sigmoid(x):
  # x = np.float16(x)
  # s = 1. / (1. + np.exp(-x))
  s = .5 * (1 + np.tanh(.5 * x))
  s[s > 1] = 1
  s[s < 0] = 0
  return s
  
def sigmoid_grad(s):
  ds = np.multiply(s, (1-s))
  return ds

def binarization(W, H):
  # [-1, 1] -> [0, 1]
  # Wb = hard_sigmoid(W / H)
  # Wb = theano.tensor.clip(W / H, -1, 1)
  Wb = W
  Wb[Wb < -1] = -1
  Wb[Wb > 1] = 1
  
  # deterministic binarization (round to the nearest)
  # Wb = theano.tensor.round(Wb)
  np.round(Wb)
  
  # 0 or 1 -> -1 or 1
  # Wb = theano.tensor.cast(theano.tensor.switch(Wb, H, -H))
  Wb[Wb < H] = -H

  return Wb

if __name__ == '__main__':

  # user specified values
  epochs = int(input("Number of epochs: "))
  hidden = int(input("Number of neurons in the hidden layer: "))
  alpha = float(input("Learning Rate (Alpha): "))

  # load data
  train_images, test_images, train_labels, test_labels = load_data()
  
  # input layer
  X = train_images

  y = train_labels

  # randomly initialize weights
  #W1 = np.random.random(X.shape[::-1])
  #W2 = np.random.random(y.shape)

  # Yann's weights initialization method
  input_range = 1.0 / X.shape[1] ** (0.5)
  W1 = np.random.normal(loc = 0, scale = input_range, size = (X.shape[1], hidden))
  W2 = np.random.uniform(size = (hidden, y.shape[1])) / np.sqrt(hidden)
  
  for i in range(epochs):
    W2 = binarization(W2, 1)
    W1 = binarization(W1, 1)

    # forward propagation
    l1 = sigmoid(np.dot(X, W1))
    l2 = sigmoid(np.dot(l1, W2))
    
    # calculate and display loss
    l2_loss = np.subtract(l2, y)
    accuracy = np.count_nonzero(l2_loss)/len(l2_loss)

    if (i%10) == 0:
      print("loss after "  + str(i) + " iterations is " + str((np.mean(np.abs(l2_loss)))))
      print("accuracy after "  + str(i) + " iterations is " + str(accuracy))

    # backward propagation
    l2_delta = np.multiply(l2_loss, sigmoid_grad(l2))
    l1_loss = np.dot(l2_delta, W2.T)
    l1_delta = np.multiply(l1_loss, sigmoid_grad(l1))
    
    # update weigths  
    W2 -= alpha*np.dot(l1.T, l2_delta)
    W1 -= alpha*np.dot(X.T, l1_delta)

