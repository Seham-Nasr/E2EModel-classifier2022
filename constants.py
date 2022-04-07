# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:27:24 2022

@author: l3s
"""

import os

DATA_AUDIO_DIR = './training_data'
TARGET_SR = 4000
OUTPUT_DIR = './output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_VAL = os.path.join(OUTPUT_DIR, 'val')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')
AUDIO_LENGTH = 10000 # 10 secs 