import numpy as np
import pandas as pd
import scipy.stats as sp
import statistics as stats
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


df = pd.read_csv('log_mini.csv')


df_ft = pd.read_csv('tf_mini.csv')


# In[4]:


df1 = df.merge(df_ft, how='left', left_on='track_id_clean', right_on='track_id')


df1.drop('track_id_clean',inplace=True,axis=1)


X = df1[ [ 'session_position', 'session_length', 'no_pause_before_play','acousticness','liveness','acoustic_vector_6']]


y = df1[['not_skipped']]


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski')  
#You can change these hyperparameters like metric etc.
knn_clf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(knn_clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

