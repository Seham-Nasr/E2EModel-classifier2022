#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
#from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier

from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from model import m5,get_data,get_data2
from keras.models import Sequential
import numpy as np
import pickle
from glob import glob
from constants import *
import sys
import keras
from data_processing import convert_data, test_dataloader
from constants import *
from keras import *
import tensorflow as tf

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    
    #__________________________________________________________________________
    convert_data(data_folder)
    num_classes = 3
    model = m5(num_classes=num_classes)

    if model is None:
        exit('Something went wrong!!')
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer= optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    #print(model.summary())
    #________________________________________________________________
    train_files = glob(os.path.join(OUTPUT_DIR_TRAIN, '**.pkl'))
    x_tr, y_tr = get_data(train_files)
    y_tr = to_categorical(y_tr, num_classes=num_classes)


    val_files = glob(os.path.join(OUTPUT_DIR_VAL, '**.pkl'))
    x_val, y_val = get_data(val_files)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # print('x_tr.shape =', x_tr.shape)
    # print('y_tr.shape =', y_tr.shape)
    # print('x_val.shape =', x_val.shape)
    # print('y_val.shape =', y_val.shape)

    #if the accuracy does not increase over 10 epochs, reduce the learning rate by half.
    #reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
    batch_size = 32
    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=8,
              verbose=1,
              shuffle=True,
              validation_data=(x_val, y_val))
    
    # Save the model.
    save_challenge_model(model_folder, model)
    
    
    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    #filename = os.path.join(model_folder, 'model.sav')
    model = keras.models.load_model(os.path.join(model_folder, "model.sav"))
    return model

def save_challenge_model(model_folder, model):
    model.save(os.path.join(model_folder, "model.sav"))

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = ['Present', 'Unknown','Absent']
    # predictions = [] 
      
    test_dataloader(recordings)     
       
    test_files = glob(os.path.join(OUTPUT_DIR_TEST, '**.pkl'))
    x_test = get_data2(test_files)
    
    predict_prob = model.predict(x_test)
    predict_classes = np.argmax(predict_prob,axis=1)
    probabilities = np.asarray(predict_prob, dtype=np.float32).max(axis=0)
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities
