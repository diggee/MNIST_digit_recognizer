# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:18:53 2020

@author: diggee
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%% reading and pre processing data files

def get_data():
    
    full_train_data = pd.read_csv('train.csv')
    full_test_data = pd.read_csv('test.csv')
    
    print(full_test_data.isnull().sum().sum())
    print(full_train_data.isnull().sum().sum())
    # there is no missing data!
    
    print((full_train_data.dtypes == object).any())
    print((full_test_data.dtypes == object).any())
    # there is no categorical data!
    
    full_train_data = shuffle(full_train_data)    
    return full_train_data, full_test_data

#%% transforming images
    
def image_transform(full_train_data, full_test_data, y):
    y = y.values.reshape(-1,1)
    X = full_train_data.reshape(-1,28,28,1)
    y = to_categorical(y)
    X_test = full_test_data.reshape(-1,28,28,1)
    return X, y, X_test

#%% scaling

def scaled_data(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    return X_train, X_test

#%% neural network
    
def neural_network(X, y, validation_split):
    model = Sequential()
    model.add(Conv2D(filters = 256, kernel_size = 4, activation = 'relu', padding = 'same', input_shape = (28,28,1), data_format = 'channels_last'))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = 4, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = 4, activation = 'relu'))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(32, 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(10, 'softmax'))
    model.summary()    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    stop_early = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
    
    train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      fill_mode='nearest',
      validation_split = validation_split
      )

    train_generator = train_datagen.flow(X, y, batch_size = 100, subset = 'training')
    valid_generator = train_datagen.flow(X, y, batch_size = 100, subset = 'validation')
    history = model.fit(train_generator, epochs = 100, verbose = 1, validation_data = valid_generator, callbacks = [stop_early, checkpoint])
    # history = model.fit(X, y, validation_split = validation_split, epochs = 100, verbose = 1, callbacks = [stop_early, checkpoint], batch_size = 100)
    return model, history

#%% plotting data

def make_plots(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])

#%% evalating the model on the test data and exporting it to csv file
    
full_train_data, full_test_data = get_data()
y = full_train_data.label
full_train_data.drop('label', axis = 1, inplace = True)
X, X_test = scaled_data(full_train_data, full_test_data)
X, y, X_test = image_transform(X, X_test, y)

model, history = neural_network(X, y, validation_split = 0.1)
make_plots(history)  

model.load_weights('model.h5')
y_pred = model.predict_classes(X_test, verbose = 2)    
df = pd.DataFrame({'ImageId':full_test_data.index+1, 'Label':y_pred})
df.to_csv('prediction.csv', index = False)