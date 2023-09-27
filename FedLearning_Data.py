#!/usr/bin/env python
# coding: utf-8

# In[236]:


from tensorflow.keras.datasets import mnist
import numpy as np
import math as m
import random
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[237]:


d = []
for i in range(10):
    d.append(train_images[train_labels == i])


# In[238]:


def get_data(nif, n_client = 4, n_class = 8):
    """non-independence factor is 0.0 to 1.0. Basically 0.0 means complete independence while 1.0 means that all the models are fed with the same data.
    Try to make n_class a multiple of n_client
    ccr is the class-to-client ratio, 
    
    returns two lists, one containing n_client training data sets, the other containing n_class label sets
    each training data set and label set is a ndarray"""
    data_set = []
    data_labels = []
    ccr = m.floor(n_class / n_client)
    alpha = nif - nif / n_client
    for i in range(n_client):
        data = np.empty((1, 28, 28))
        label = np.array([])
        for j in range(n_class):
            if (j > ccr * i - 1 and j < ccr * i + ccr):
                n_data = round((1 - alpha) * d[j].shape[0])
                index = random.sample(sorted(range(d[j].shape[0])), n_data)
            else:
                n_data = round(alpha / (n_client - 1) * d[j].shape[0])
                index = random.sample(sorted(range(d[j].shape[0])), n_data)
            data = np.append(data, d[j][index], 0)
            label = np.append(label, j * np.ones(n_data))
        data_set.append(data[1:])
        data_labels.append(label)
    return (data_set, data_labels)

