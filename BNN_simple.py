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

def load_data(load_keras=False, method_one=True, method_two=False):
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
  
  if (method_one):
    train_data = pd.read_csv("mnist_training.csv")
    
    images = train_data.iloc[:,1:].values
    images = images.astype(np.float)
    images = np.multiply(images, 1.0/255.0)
    image_size = images.shape[1]
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    
    # print(train_data.head(5))
    
    labels_flat = train_data[['label']].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    num_labels =labels_flat.shape[0]
    index_offset = np.arange(num_labels) * labels_count
    labels_one_hot = np.zeros((num_labels, labels_count))
    labels_one_hot.flat[index_offset + labels_flat.ravel()] = 1
    labels = labels_one_hot.astype(np.uint8)
    
    train_images = images[:5000]
    train_labels = labels[:5000]
    
    test_images = images[6000:7000]
    test_labels = labels[6000:7000]
      
  if (method_two):
    # manually load data
    train_data = pd.read_csv("mnist_training.csv")

    for column in train_data:
      train_data[column] = pd.to_numeric(train_data[column], errors="coerce").fillna(0)

    # label manually loaded training data
    images = train_data.iloc[:10000,1:]
    labels = train_data.iloc[:10000,:1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size = 0.2, random_state=25)
  
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
  s = .5 * (1 + np.tanh(.5 * x))
  # s[s > 1] = 1
  # s[s < 0] = 0
  return s
  
def sigmoid_grad(s):
  ds = np.multiply(s, (1-s))
  return ds

def binarization(W):
  Wb[Wb < -1] = -1.0
  Wb[Wb > 1] = 1.0
  
  Wb[Wb > 0.1 and Wb < 1] = 0.1
  Wb[Wb < -0.1 and Wb > -1] = -0.1

  return Wb

if __name__ == '__main__':

  # user specified values
  epochs = int(input("Number of epochs: "))
  hidden = int(input("Number of neurons in the hidden layer: "))
  alpha = float(input("Learning Rate: "))
  
  # load data
  train_images, test_images, train_labels, test_labels = load_data()
  
  print("-----------------------------------")
  
  X = train_images
  y = train_labels

  # randomly initialize weights
  #W1 = np.random.random(X.shape[::-1])
  #W2 = np.random.random(y.shape)

  # Yann's weights initialization method
  input_range = 1. / X.shape[1] ** (.5)
  W1 = np.random.normal(loc = 0, scale = input_range, size = (X.shape[1], hidden))
  W2 = np.random.uniform(size = (hidden, y.shape[1])) / np.sqrt(hidden)

  # training
  for i in range(epochs):
    # W2 = binarization(W2)
    # W1 = binarization(W1)

    # forward propagation
    l1 = sigmoid(np.dot(X, W1))
    l2 = sigmoid(np.dot(l1, W2))
    
    # calculate and display loss
    l2_loss = np.subtract(l2, y) 
    accuracy = np.mean(np.argmax(l2, axis=1) == y)

    if (i%10) == 0:
      print("training loss and accuracy after "  + str(i) + " iterations are " + str((np.mean(np.abs(l2_loss)))) + " and " + str(accuracy))
            
    # backward propagation
    l2_delta = np.multiply(l2_loss, sigmoid_grad(l2))
    l1_loss = np.dot(l2_delta, W2.T)
    l1_delta = np.multiply(l1_loss, sigmoid_grad(l1))
    
    # update weigths  
    W2 += -alpha*np.dot(l1.T, l2_delta)
    W1 += -alpha*np.dot(X.T, l1_delta)
    
X = test_images
y = test_labels

l1 = sigmoid(np.dot(X, W1))
l2 = sigmoid(np.dot(l1, W2))
print("-----------------------------------")
print("shape of y: " + str(y.shape))
print("shape of l2: " + str(l2.shape))
print("-----------------------------------")
print(y[3:7])
print(np.round(l2[3:7], 1))
# print(np.argmax(y, axis=1))
# print(np.argmax(l2, axis=1))
print("-----------------------------------")
print("number of nonzero elements in y and l2: %d and %d" % (np.count_nonzero(y), np.count_nonzero(l2)))
print("testing accuracy: %.12f" % np.mean(np.argmax(l2, axis=1) == y))
print("-----------------------------------")
l2[np.where(l2==np.max(l2))]=1
l2[np.where(l2!=np.max(l2))]=0
print(y[:3])
print(l2[:3])
total = 0
correct = 0
for i in range(len(l2)):
  total+=1
  if np.equal(l2[i], y[i]).all():
    correct+=1
print("testing accuracy: " + str(correct*100/total))
