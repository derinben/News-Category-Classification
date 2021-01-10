from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import dataQA

df = dataQA.df.copy()

#Data Tidying
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

df['desc'] = df['headline'].astype(str)+"-"+df['short_description']
df.drop(columns =['headline','short_description'],axis = 1, inplace=True)
df.astype(str)
classes = df['category'].value_counts().index

#Splitting the data into train, validation and test data
X,Y = df['desc'],df['category']

#80% to train , 10% for validation , 10% for testing
X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size=0.2, random_state=42)
X_val, X_test , y_val, y_test= train_test_split(X_val,y_val, test_size=0.5, random_state=42)

#Tokenizing the data

vocab_size =20000
max_length = 150
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size,lower=True, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index_items = tokenizer.word_index.items()

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train,maxlen= max_length,padding=padding_type, truncating=trunc_type)
y_train = np.asarray(y_train)
y_train = pd.get_dummies(y_train)

X_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val,maxlen= max_length,padding=padding_type, truncating=trunc_type)
y_val = np.asarray(y_val)
y_val = pd.get_dummies(y_val)

train_set = np.array(X_train)
val_set = np.array(X_val)

train_label = np.array(y_train)
val_label = np.array(y_val)

y_test = pd.get_dummies(y_test)
y_test = np.asarray(y_test)
y_test = np.argmax(y_test,axis=1)   #Ground truth label for testing and evaluating

print("Train Set Shape:",train_set.shape)
print("Train Labels Shape:",train_label.shape)
print("Validation Set Shape:",val_set.shape)
print("Validation Labels Shape:",val_label.shape)

