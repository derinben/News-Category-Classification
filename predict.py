import tokenizer
import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def prediction(inference_data):
    X = tokenizer.tokenizer.texts_to_sequences(inference_data)
    X = pad_sequences(X, maxlen=tokenizer.max_length, padding=tokenizer.padding_type, truncating=tokenizer.trunc_type)
    pred = model.model.predict(X)
    pred_value = tf.argmax(pred,axis =1).numpy()
    return pred_value


#y_pred = prediction(X_test)

#Classification report
def cf_matrix(y_test,y_pred):
    print(classification_report(np.asarray(y_test), np.asarray(y_pred)))
    cf_matrixrep = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,10))
    heatmap = sns.heatmap(cf_matrixrep, xticklabels=tokenizer.classes,
                          yticklabels=tokenizer.classes,
                          annot=True, fmt='d', color='blue')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('Confusion matrix of model')

#cf_matrix(y_test,y_pred)