# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:27:24 2022

@author: l3s
"""


from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from tensorflow.keras import backend as K

from keras.models import Sequential
import numpy as np
import pickle
from glob import glob
from constants import *
from data_processing import convert_data
import sys
from data_processing import convert_data

#_____________________________________________________________


def m5(num_classes=3):
    print('Using Model M5')
    m = Sequential()
    m.add(Conv1D(128,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(128,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(256,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: K.mean(x, axis=1))) 
    m.add(Dense(num_classes, activation='softmax'))
    return m


def get_data(file_list, progress_bar=False):
    def load_into(_filename, _x, _y):
        with open(_filename, 'rb') as f:
            audio_element = pickle.load(f)
            _x.append(audio_element['audio'])
            _y.append(int(audio_element['class_id']))

    x, y = [], []
    for filename in file_list:
        load_into(filename, x, y)
    return np.array(x), np.array(y)

def get_data2(file_list, progress_bar=True):
    def load_into(_filename, _x):
        with open(_filename, 'rb') as f:
            audio_element = pickle.load(f)
            _x.append(audio_element['audio'])
          

    x, y = [], []
    for filename in file_list:
        load_into(filename, x)
    return np.array(x)

# if __name__ == '__main__':

    

    # Parse the arguments.
    # if not (len(sys.argv) == 3 or len(sys.argv) == 4):
    #     raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')
    
    # Define the data and model foldes.
    # data_folder = sys.argv[1]
    # #model_folder = sys.argv[2]
    
    # # Change the level of verbosity; helpful for debugging.
    # if len(sys.argv)==4 and is_integer(sys.argv[3]):
    #     verbose = int(sys.argv[3])
    # else:
    #     verbose = 1   
    
    # convert_data(data_folder)
    ##########################################################
    # num_classes = 3
    # model = m5(num_classes=num_classes)

    # if model is None:
    #     exit('Something went wrong!!')

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # #print(model.summary())
    
    # #________________________________________________________________
    
    # convert_data(iglob(os.path.join(DATA_AUDIO_DIR, '**/**.wav'), recursive=True))
    # #________________________________________________________________
    # train_files = glob(os.path.join(OUTPUT_DIR_TRAIN, '**.pkl'))
    # x_tr, y_tr = get_data(train_files)
    # y_tr = to_categorical(y_tr, num_classes=num_classes)

    # test_files = glob(os.path.join(OUTPUT_DIR_TEST, '**.pkl'))
    # x_te, y_te = get_data(test_files)
    # y_te = to_categorical(y_te, num_classes=num_classes)

    # # print('x_tr.shape =', x_tr.shape)
    # # print('y_tr.shape =', y_tr.shape)
    # # print('x_te.shape =', x_te.shape)
    # # print('y_te.shape =', y_te.shape)

    # # if the accuracy does not increase over 10 epochs, reduce the learning rate by half. In the coming experiments
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    # batch_size = 128
    # model.fit(x=x_tr,
    #           y=y_tr,
    #           batch_size=batch_size,
    #           epochs=10,
    #           verbose=1,
    #           shuffle=True,
    #           validation_data=(x_te, y_te),
    #           callbacks=[reduce_lr])
