#!/usr/bin/env python3

import numpy as np
import random as rd
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# deterministic behaviour
np.random.seed(1024)

def load_data():
  # manually load data
  train_data = pd.read_csv("mnist_training.csv")

  for column in train_data:
    train_data[column] = pd.to_numeric(train_data[column], errors="coerce").fillna(0)

  # label manually loaded training data
  images = train_data.iloc[:10000,1:]
  labels = train_data.iloc[:10000,:1]
  train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size = 0.2, random_state=42)
  
  # switch from pandas data frames to numpy arrays
  train_images = train_images.values
  test_images = test_images.values
  train_labels = train_labels.values.ravel()
  test_labels = test_labels.values.ravel()

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
  s = .5 * (1.0 + np.tanh(.5 * x))
  # s[s > 1] = 1
  # s[s < 0] = 0
  return s
  
def sigmoid_grad(s):
  ds = np.multiply(s, (1.0-s))
  return ds

def softmax(x):
  orig_shape = x.shape
  print(orig_shape)
  if len(x.shape) > 1:
    # Matrix
    c=x.max(1).reshape(-1,1)
    x=np.exp(np.subtract(x, c))
    x=x/(x.sum(1).reshape(-1,1))
  else:
    # Vector
    c=np.max(x)
    x=np.exp(np.subtract(x, c))
    x=np.divide(x, x.sum())
  return x
  
def stable_softmax(X):
  exps = np.exp(X - np.max(X, axis=1, keepdims=True))
  return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(X, y):
  m = y.shape[0]
  p = X
  # smoothing factor to prevent 'Divide by zero in log' error
  smooth_factor = 1e-14
  log_likelihood = -np.log(p[np.arange(m), np.argmax(y, axis=1)] + smooth_factor)
  loss = np.sum(log_likelihood) / m
  return loss

def add_variation(preds, variation):
  p = preds.copy()
  v = np.tile(variation, p.shape)
  p = np.array(p)
  v = np.array(v)
  return np.add(p, p*v)

def binarization(W):
  Wb = W
  Wb[Wb < -1] = -1.0
  Wb[Wb > 1] = 1.0
  
  Wb[np.logical_and(Wb > 0.1, Wb < 1)] = 0.1
  Wb[np.logical_and(Wb < -0.1, Wb > -1)] = -0.1

  return Wb

if __name__ == '__main__':

  # user specified values
  epochs = int(input("Number of epochs: "))
  hidden = int(input("Number of neurons in the hidden layer: "))
  alpha = float(input("Learning Rate: "))
  
  # load data
  train_images, test_images, train_labels, test_labels = load_data()
  print("-----------------------------------")
  print('Rows: %d, columns: %d in the training set' % (train_images.shape[0], train_images.shape[1]))
  
  X = train_images
  y = train_labels

  # randomly initialize weights
  # Yann's weights initialization method
  input_range = 1. / X.shape[1] ** (.5)
  W1 = np.random.normal(loc = 0, scale = input_range, size = (X.shape[1], hidden))
  W2 = np.random.uniform(size = (hidden, y.shape[1])) / np.sqrt(hidden)

  # training
  for i in range(epochs):
    W2 = binarization(W2)
    W1 = binarization(W1)

    # forward propagation
    l1 = sigmoid(np.dot(X, W1))
    l2 = stable_softmax(np.dot(l1, W2))

    # calculate and display loss
    l2_loss = cross_entropy(l2, y)
    accuracy = np.mean(np.argmax(l2, axis=1) == np.argmax(y, axis=1))

    if (i%100) == 0:
      print("training loss and accuracy after "  + str(i) + " iterations are " + str((np.mean(np.abs(l2_loss)))) + " and " + str(accuracy*100))
            
    # backward propagation
    # l2_delta = delta_cross_entropy(l2_loss, y)
    # l1_loss = np.dot(l2_delta, W2.T)
    l2_delta = l2
    y_copy = y.copy()
    l2_delta[np.arange(X.shape[0]), np.argmax(y_copy, axis = 1)] -= 1
    l2_delta /= X.shape[0]
    
    l1_loss = np.dot(l2_delta, W2.T)
    l1_delta = np.multiply(l1_loss, sigmoid_grad(l1))
    
    # update weigths    
    W2 += -alpha*np.dot(l1.T, l2_delta)
    W1 += -alpha*np.dot(X.T, l1_delta)
    
X = test_images
y = test_labels

y_copy_one = y.copy()
y_copy_two = y.copy()
y_copy_three = y.copy()

l1 = sigmoid(np.dot(X, W1))
l2 = stable_softmax(np.dot(l1, W2))
l2_copy = l2.copy()

print("-----------------------------------")

print("shape of y: " + str(y.shape))
print("shape of l2: " + str(l2.shape))

print("-----------------------------------")

print(np.argmax(l2_copy, axis=1))
print(np.argmax(y_copy_one, axis=1))

print("-----------------------------------")

print("number of nonzero elements in y and l2: %d and %d" % (np.count_nonzero(y), np.count_nonzero(l2)))
print("testing accuracy: %.12f" % (np.mean(np.argmax(l2_copy, axis=1) == np.argmax(y_copy_two, axis=1))*100))

print("-----------------------------------")

print(y[10:13])
print(l2[10:13])

print("-----------------------------------")

var = rd.choice([0.1, -0.1])

l2_copy_variable  = add_variation(l2_copy, var)
print("prediction scores with variation of " + str(var))
print(l2_copy_variable[10:13])

print("-----------------------------------")

total = 0
correct = 0
for i in range(len(l2)):
  total+=1
  if np.equal(np.argmax(l2_copy_variable, axis=1)[i], np.argmax(y_copy_one, axis=1)[i]).all():
    correct+=1
print("testing accuracy: " + str(correct*100/total))
