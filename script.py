# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
edges=pd.read_csv('data/training_set.txt',header=None,sep=' ')
edges_test=pd.read_csv('data/testing_set.txt',header=None,sep=' ')

node_info=pd.read_csv('data/node_information.csv',header=None)
article2id=dict(enumerate(np.unique(node_info[0])))
id2article = {v: k for k, v in article2id.iteritems()}

#import scipy.sparse as sp
#from tqdm import tqdm
edge_w_index=edges
edge_w_index.iloc[:,0]=edge_w_index.iloc[:,0].map(lambda x: id2article[x])
edge_w_index.iloc[:,1]=edge_w_index.iloc[:,1].map(lambda x: id2article[x])

edge_w_index_test=edges_test
edge_w_index_test.iloc[:,0]=edge_w_index_test.iloc[:,0].map(lambda x: id2article[x])
edge_w_index_test.iloc[:,1]=edge_w_index_test.iloc[:,1].map(lambda x: id2article[x])
#node_info.iloc[:,0]=node_info.iloc[:,0].map(lambda x: id2article[x])
node_info.columns=['article_id','year','title','author','class','abstract']

#adj_matrix=sp.lil_matrix((len(node_info),len(node_info)))
#data_np=np.array(data)
#for row in tqdm(data_np):
#    adj_matrix[row[0],row[1]]=row[2]

from gensim.models import Doc2Vec



from collections import namedtuple
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for row in range(len(node_info)):
    words = node_info['abstract'][row].lower().split()
    tags = [row]
    docs.append(analyzedDocument(words, tags))
doc2vect_abstract=Doc2Vec(docs , size=100, window=8, min_count=5, workers=4)
array_abstract=np.array([doc2vect_abstract[i] for i in range(len(node_info))])

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for row in range(len(node_info)):
    words = node_info['title'][row].lower().split()
    tags = [row]
    docs.append(analyzedDocument(words, tags))
doc2vect_title=Doc2Vec(docs , size=20, window=2, min_count=5, workers=4)
array_title=np.array([doc2vect_title[i] for i in range(len(node_info))])


LINE_emb=pd.read_csv('data/LINE_emb.txt',header=None,sep=" ",skiprows=1)
LINE_emb=np.array(LINE_emb.sort_values([0]).iloc[:,:113])
nodes_matrix=np.concatenate((LINE_emb,np.array(node_info.iloc[:,1]).reshape((-1,1)),array_title,array_abstract),1)


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





from keras.models import Input, Model
from keras.layers import Dense,Concatenate,Dropout
n_features=nodes_matrix.shape[1]-1
input_A=Input(shape=(n_features,), name='node_A')
input_B=Input(shape=(n_features,), name='node_B')
x=Concatenate()([input_A,input_B])
x = Dropout(0.25)(x)
x=Dense(256,activation='relu')(x)
x =Dropout(0.25)(x)
x=Dense(512,activation='relu')(x)
x =Dropout(0.25)(x)
y=Dense(1,activation='sigmoid')(x)

VAL_PROP=0.25
edge_w_index=np.array(edge_w_index)
np.random.shuffle(edge_w_index)
N_VAL=int(edge_w_index.shape[0]*VAL_PROP)
egdes_val=edge_w_index[:N_VAL,:]
egdes_train=edge_w_index[N_VAL:,:]

model = Model(inputs=[input_A, input_B], outputs=y)
model.compile(optimizer='nadam',
              loss='binary_crossentropy',metrics=['accuracy',f1])
normalized_nodes_matrix=(nodes_matrix-np.mean(nodes_matrix,0))/np.std(nodes_matrix,0)
model.fit_generator(generator=generate_nodes_pair(normalized_nodes_matrix,egdes_train), validation_data=generate_nodes_pair(normalized_nodes_matrix,egdes_val),
                    validation_steps=500,steps_per_epoch=2000,epochs=10)
set_kaggle=generate_kaggle(normalized_nodes_matrix,edge_w_index_test)
pred=(model.predict(set_kaggle)>0.5).astype(int)
pred_df=pd.DataFrame(pred).to_csv('submission4.csv')



        
