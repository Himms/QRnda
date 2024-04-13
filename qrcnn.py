import warnings
import os

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import spacy

import numpy as np 
import pandas as pd 
import io
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

warnings.simplefilter('ignore')
nlp=spacy.load('en_core_web_sm')

from google.colab import files
uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['dataset_B_05_2020 (1).csv']))
data.head()
data.shape

#Encoding 'status' as label 1 & 0 , naming the field as target
data['target'] = pd.get_dummies(data['status'])['legitimate'].astype('int')
data.drop('status',axis = 1, inplace=True)
data.head()

from sklearn.model_selection import train_test_split
#Train & Test Set
X= data.iloc[: , 1:-1]
#y = upsampled_df['Churn']
y= data['target']

train_x,test_x,train_y,test_y = \
train_test_split(X,y, stratify=y, train_size=0.75)
print(train_x.shape)

print("\n--Training data samples--")
train_x = np.expand_dims(train_x, axis=2)
test_x = np.expand_dims(test_x, axis=2)
print(train_x.shape)
input_shape = train_x[1].shape

print(input_shape)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape= input_shape),
    layers.Conv1D(filters=64,kernel_size=7,activation='relu'),
    layers.BatchNormalization(),
    #layers.Conv1D(filters=32,kernel_size=3,activation='relu'),
    #layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2) ,
    layers.Dropout(0.3),
    layers.Flatten(), # flatten out the layers
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.0001,
    restore_best_weights=True,
)
model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping],
)

model.evaluate(test_x, test_y)

# Get model predictions
predictions = model.predict(test_x)


for i in range(len(predictions)):
  if predictions[i][0] >= 0.5:
    predictions[i][0] = int(1)
  else:
    predictions[i][0] = 0
 # print(predictions[0])




# If your predictions are one-hot encoded, convert them to integer labels
#predicted_labels = np.argmax(predictions, axis=1)
predicted_labels = predictions.astype(int)
print(predicted_labels)

test_integer_labels = test_y.astype(int).tolist()
print(test_integer_labels)

import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *


conf_matrix = confusion_matrix(test_integer_labels, predicted_labels)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
sns.heatmap(conf_matrix, annot=True)
print("Precision of the CNN :\t"+str(precision_score(test_integer_labels, predicted_labels)))
print("Recall of the CNN    :\t"+str(recall_score(test_integer_labels, predicted_labels)))
print("F1 Score of the Model :\t"+str(f1_score(test_integer_labels, predicted_labels)))
print("ACC Score of the Model :\t"+str(accuracy_score(test_integer_labels, predicted_labels)))

