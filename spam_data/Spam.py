
# coding: utf-8

# # import glob for iterating files in folder and use dataframe for structuring

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import pandas as pd

body=[]
list_of_files = glob.glob('./spam_data/0*.txt')

for fileName in list_of_files:
    
    file = open(fileName, encoding = "ISO-8859-1")
    body.append(file.read())
    
    file.close()

body_file = pd.DataFrame(data = body,columns = ['post'])


label_file = pd.read_csv('./spam_data/labels.txt',header = None,delimiter=" ",names=["label",'Name'])



import itertools
import os

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, Dropout
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from tensorflow.contrib.keras.python.keras import utils

# This code was tested with TensorFlow v1.3
print("You have TensorFlow version", tf.__version__)


train_size = int(len(body_file) * .7)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(body_file) - train_size))

np.random.seed()
train_posts = body_file['post'][:train_size]
train_tags = label_file['label'][:train_size]

test_posts = body_file['post'][train_size:]
test_tags = label_file['label'][train_size:]

len(train_posts), len(train_tags)



len(test_posts), len(test_tags)



max_words = 100
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


tokenize.fit_on_texts(train_posts) # only fit on train


text = "tr"

tokenize.texts_to_matrix([text])



x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)



from scipy.spatial import distance
def eucledean(a,b):
        return distance.euclidean(a,b)
    
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


from collections import Counter

#Custom Classifier
class myCustomClassifier():
    def __init__(self,n_number = 3):
        self.n_number = n_number
    
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def closest(self,row):
        
        tempDist = []
        tempFull = []
       
        counter = 0
        
        for i in range(1,len(x_train)):
            
            dist = eucledean(row,self.x_train[i])
            tempDist.append((dist,self.y_train[i]))
           
        
        tempFull = [i[1] for i in sorted(tempDist)[:self.n_number]]
        
        voteResult = Counter(tempFull).most_common(1)[0][0]
       
        # Takes votes of k - number and returns label which has most votes          
        return voteResult
    
    
        
    def predict(self,x_test):
        
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
#Implementing custom classifier
knn = myCustomClassifier(3)
knn.fit(x_train, y_train) 


Final_predictions = knn.predict(x_test)


from sklearn.metrics import accuracy_score
print("Accuracy :- ",accuracy_score(y_test,Final_predictions))


