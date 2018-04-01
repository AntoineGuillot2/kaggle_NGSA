import pandas as pd
import numpy as np
from utils import *



##Reading egdes data
edges=pd.read_csv('data/training_set.txt',header=None,sep=' ')
edges_test=pd.read_csv('data/testing_set.txt',header=None,sep=' ')


##Reading nodes info
node_info=pd.read_csv('data/node_information.csv',header=None)
article2id=dict(enumerate(np.unique(node_info[0])))
id2article = {v: k for k, v in article2id.iteritems()}
node_info.columns=['article_id','year','title','author','class','abstract']


##Add index to the edges to be able to create the batches on the fly
edge_w_index=edges
edge_w_index.iloc[:,0]=edge_w_index.iloc[:,0].map(lambda x: id2article[x])
edge_w_index.iloc[:,1]=edge_w_index.iloc[:,1].map(lambda x: id2article[x])

edge_w_index_test=edges_test
edge_w_index_test.iloc[:,0]=edge_w_index_test.iloc[:,0].map(lambda x: id2article[x])
edge_w_index_test.iloc[:,1]=edge_w_index_test.iloc[:,1].map(lambda x: id2article[x])

##Doc2Vec representation of the abstract and of the title
from gensim.models import Doc2Vec
from collections import namedtuple

###Abstract
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for row in range(len(node_info)):
    words = node_info['abstract'][row].lower().split()
    tags = [row]
    docs.append(analyzedDocument(words, tags))
doc2vect_abstract=Doc2Vec(docs , size=100, window=8, min_count=5, workers=4)
array_abstract=np.array([doc2vect_abstract[i] for i in range(len(node_info))])



###Title
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for row in range(len(node_info)):
    words = node_info['title'][row].lower().split()
    tags = [row]
    docs.append(analyzedDocument(words, tags))
doc2vect_title=Doc2Vec(docs , size=20, window=2, min_count=5, workers=4)
array_title=np.array([doc2vect_title[i] for i in range(len(node_info))])


##Read the line embedding computed in the LINE folder
LINE_emb=pd.read_csv('data/LINE_emb.txt',header=None,sep=" ",skiprows=1)
LINE_emb=np.array(LINE_emb.sort_values([0]).iloc[:,:113])

###Representation of the nodes as the concatenation of tile+abstract+node_embeeing+year
nodes_matrix=np.concatenate((LINE_emb,np.array(node_info.iloc[:,1]).reshape((-1,1)),array_title,array_abstract),1)


###Neural network
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


##Selection of validation proportion
##Build test and train dataset
VAL_PROP=0.25
edge_w_index=np.array(edge_w_index)
np.random.shuffle(edge_w_index)
N_VAL=int(edge_w_index.shape[0]*VAL_PROP)
egdes_val=edge_w_index[:N_VAL,:]
egdes_train=edge_w_index[N_VAL:,:]


model = Model(inputs=[input_A, input_B], outputs=y)
model.compile(optimizer='nadam',
              loss='binary_crossentropy',metrics=['accuracy',f1])

##Normalize the nodes to zeros means and unit variance
normalized_nodes_matrix=(nodes_matrix-np.mean(nodes_matrix,0))/np.std(nodes_matrix,0)

##Fit the model
model.fit_generator(generator=generate_nodes_pair(normalized_nodes_matrix,egdes_train), validation_data=generate_nodes_pair(normalized_nodes_matrix,egdes_val),
                    validation_steps=500,steps_per_epoch=2000,epochs=10)


##Prediction on the score dataset
set_kaggle=generate_kaggle(normalized_nodes_matrix,edge_w_index_test)
pred=(model.predict(set_kaggle)>0.5).astype(int)
pred_df=pd.DataFrame(pred).to_csv('submission4.csv')
