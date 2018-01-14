# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:48:00 2018

@author: Sagar Lakhmani
"""

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM,GRU
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
from keras import optimizers
from keras import regularizers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

img_rows,img_cols,img_depth = 32,32,30
num_samples = int(len(image_list)/img_depth)
num_classes = 10
X_t = np.resize(image_list,(num_samples,img_rows,img_cols,img_depth))
y_t = np.asarray(np.resize(y_tr,(num_samples,img_depth)))
train_data = [X_t,y_t]

(X_train, y_train) = (train_data[0],train_data[1])

train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))
#train_set = []
for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h][:][:][:]
    
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

model = Sequential()
model.add(Convolution3D(32,(5,5,5),strides = (1,1,1), input_shape=(1, img_rows, img_cols, img_depth), activation='relu',data_format='channels_first'))

model.add(MaxPooling3D((3,3,3), data_format='channels_first'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128,kernel_initializer='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,kernel_initializer='normal'))

model.add(Activation('softmax'))
#model.add(TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1),
#            activation='relu', padding='same'), input_shape=(1, img_rows, img_cols, img_depth)))
#model.add(TimeDistributed(Conv2D(64, (5,5),
#            kernel_initializer="he_normal", activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(1, 1))))
#
#model.add(TimeDistributed(Flatten()))
#model.add(GRU(128, dropout=0.5))
#model.add(Dense(num_classes, activation='softmax'))


sgd = optimizers.RMSprop(lr=0.0001,rho = 0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

#Y_train = to_categorical(y_train[:,0], num_classes=10) # convert to one hot encoding
Y_train = np.zeros([num_samples,num_classes])
for i in range(num_samples):
    for j in range(num_classes):
        if y_train[i][j]==j+1:
            Y_train[i][j] = 1
        else:
            Y_train[i][j] = 0

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.25, random_state=4)


# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new), 
          batch_size=1,epochs = 100,shuffle=True)

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(100)

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
model.save('Modelv2.h5')
