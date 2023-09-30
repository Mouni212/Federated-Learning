#!/usr/bin/env python
# coding: utf-8

# In[148]:


from tensorflow.keras.datasets import mnist
import numpy as np
import math as m
import random


# In[ ]:


import numpy as np


# In[114]:


def get_general(labels, alpha, n_client, nrlabels):
    #getting the number of pieces of data that is general to the clients
    n_general = {}
    for i in nrlabels:
        n_general[i] = round(alpha / (n_client - 1) * np.sum(labels == i))
    return n_general


# In[146]:


def get_labels(nif, labels, assignment, seed = None):
    """returns a list of ndarrays, the nth list contains 0s and 1s to show which data are selected for the (n+1)th client.
    
    nif is the none independence factor. 0 means that the datasets returned for each client are completely independent from each other,
    while an nif of 1 means that all the clients have datasets from the same data distributions. for example, when nif is 0 and we have 
    2 clients and four classes, the first client will only have data from two classes, while the second one only has data from the other
    two classes. if it takes on some value between 0 and 1: each dataset will contain more of the classes they are assigned in assignment,
    and some of the other classes.
    
    labels is the (either ndarray or list) of the labels in the training dataset.
    
    assignment is a (either a list of lists or ndarray) containing the labels for emphasized datasets of each client.
    eg. [[0, 1], [2, 3]] or [['cats', 'dogs'], ['frogs', 'pandas']]. these will be the datasets assigned to each client if nif equaled 0.
    the clients will be fed more data for these classes unless nif == 1.
    
    seed is the seed used for random functions, to make sure that identical results are obtainable.
    """
    #getting the length of labels and the number of clients
    if (type(labels) == list):
        labels = np.array(labels)
    n_data = labels.shape[0]
    assignment = np.array(assignment)
    
    
    #the different elements in labels
    if (type(assignment) == list):
        assignment = np.array(assignment)
    n_client = assignment.shape[0]
    nrlabels = assignment.reshape(-1)
    alpha = nif - nif / n_client
    n_general = get_general(labels, alpha, n_client, nrlabels)
    results = []
    for i in range(n_client):
        results.append(np.zeros_like(labels))
    
    
    #set the random seed
    if (seed != None):
        np.random.seed(seed)
    
    
    #assign data without repitition by shuffling 
    for i in nrlabels:
        temp = np.array(np.where(labels == i)[0])
        np.random.shuffle(temp)        
        for j in range(n_client):
            results[j][temp[j * n_general[i] : (j + 1) * n_general[i]]] = 1
            if (np.in1d(i, assignment[j])):
                results[j][temp[n_client * n_general[i]:]] = 1
    return results

