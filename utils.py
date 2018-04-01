#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:34:30 2018

@author: antoine
"""
import numpy as np

##Genereates nodes pair from an edge pair with a label
##used to train the FCN
def generate_nodes_pair(nodes_matrix,edges_w_index,batch_size=256):
    while True:
        n_features=nodes_matrix.shape[1]-1
        edges_idx=np.random.randint(edges_w_index.shape[0],size=batch_size).tolist()
        selected_edges=np.array(edges_w_index)[edges_idx]
        batch_A=np.zeros((batch_size,n_features))
        batch_B=np.zeros((batch_size,n_features))
        batch_y=np.zeros((batch_size,))
        i=0
        for edge in selected_edges:
            batch_A[i]+=nodes_matrix[selected_edges[i][0],1:]
            batch_B[i]+=nodes_matrix[selected_edges[i][1],1:]
            batch_y[i]+=selected_edges[i][2]
            i+=1
        yield {'node_A':batch_A,'node_B':batch_B},batch_y
        
##Genereates nodes pair from an edge pair 
##used to make prediction using the neural network   
def generate_kaggle(nodes_matrix,edges_w_index,batch_size=256):
    n_features=nodes_matrix.shape[1]-1
    edges_idx=list(range(edges_w_index.shape[0]))
    selected_edges=np.array(edges_w_index)[edges_idx]
    batch_A=np.zeros((len(edges_idx),n_features))
    batch_B=np.zeros((len(edges_idx),n_features))
    i=0
    for edge in selected_edges:
        batch_A[i]+=nodes_matrix[selected_edges[i][0],1:]
        batch_B[i]+=nodes_matrix[selected_edges[i][1],1:]
        i+=1
    return {'node_A':batch_A,'node_B':batch_B}
        
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))