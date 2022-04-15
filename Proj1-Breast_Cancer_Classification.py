# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:08:17 2022

@author: hong9
"""

#Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os

#1. Read CSV data
filePath = r"C:\Users\hong9\Documents\SHRDC\ML_learningMaterials\04-DeepLearning\Projects\Project_1\data.csv"
breastCancer = pd.read_csv(filePath)

#2. Preprocessing the data (remove unused data)
breastCancer = breastCancer.drop(['id','Unnamed: 32'], axis=1)

#3. Split data into features and label
breastCancer_features = breastCancer.copy()
breastCancer_label = breastCancer_features.pop('diagnosis')
# Printing the features and label
print('----- ----- Features ----- -----')
print(breastCancer_features.head())
print('----- ----- Label ----- -----')
print(breastCancer_label.head())


#%%
#4. One hot encode label
breastCancer_label_OH = pd.get_dummies(breastCancer_label)
# Printing the one hot label
print('----- ----- One hot Label ----- -----')
print(breastCancer_label_OH.head())

#5. Split the features and labels into train-validation-test sets (using 60:20:20 split)
SEED = 12345
x_train, x_iter, y_train, y_iter = train_test_split(breastCancer_features, breastCancer_label_OH, test_size=0.4, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter, y_iter, test_size=0.5, random_state=SEED)

#6. Normalize the features and fit with training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Data preparation is completed

#%%
#7. Create a Feedforward Neural Network Model using TensorFlow Keras
number_input = x_train.shape[-1]
number_output = y_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input)) 
model.add(tf.keras.layers.Dense(64,activation='elu'))
model.add(tf.keras.layers.Dense(32,activation='elu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(number_output,activation="softmax"))

#8. Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#%%
#9. Train and evaluate model
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\hong9\Documents\SHRDC\ML_learningMaterials\04-DeepLearning\Projects\Tensorboard\proj1_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE=32
history = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tb_callback,es_callback])

#%%
#Evaluate with test data for wild testing
test_result = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")



