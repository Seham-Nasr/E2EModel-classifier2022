# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:59:37 2022

@author: l3s
"""

import pickle
from glob import iglob
import numpy as np
import librosa
from shutil import rmtree
from constants import *
import pandas as pd
import re
import sys
from helper_code import *
#______________________________________________

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass





class_ids = {
    'Present': 0,
    'Unknown': 1,
    'Absent': 2,
}


def extract_class_id(wav_filename):
    if 'Unknown' in wav_filename:
        return class_ids.get('Unknown')
    elif 'Present' in wav_filename:
        return class_ids.get('Present')
    else:
        return class_ids.get('Absent')


def read_audio_from_filename(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def convert_data(data_folder):
    
   del_folder(OUTPUT_DIR_TRAIN)
   del_folder(OUTPUT_DIR_VAL)
   mkdir_p(OUTPUT_DIR_TRAIN)
   mkdir_p(OUTPUT_DIR_VAL)
   
   for i, wav_filename in enumerate(iglob(os.path.join(data_folder, '**/**.wav'), recursive=True)):
        #_________________________________________Extract sound class
        # # Find patient data files.
        # def find_patient_files(data_folder):
        #     # Find patient files.
        #     filenames = list()
        #     for f in sorted(os.listdir(data_folder)):
        #         root, extension = os.path.splitext(f)
        #         if not root.startswith('.') and extension=='.txt':
        #             filename = os.path.join(data_folder, f)
        #             filenames.append(filename)

        #     # To help with debugging, sort numerically if the filenames are integers.
        #     roots = [os.path.split(filename)[1][:-4] for filename in filenames]
        #     if all(is_integer(root) for root in roots):
        #         filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

        #     return filenames
        d = wav_filename.replace("wav", "txt")
        #d = wav_filename[:-7:]
        d = d.replace("_TV", "")
        d = d.replace("_AV", "")
        d = d.replace("_MV", "")
        d = d.replace("_PV", "")
        d = d.replace("_1", "")
        d = d.replace("_2", "")
        d = d.replace("_Phc", "")
        d = d.replace("_3", "")
                 
        Class_i = "#Murmur: "
        d.replace("_", "")
        with open(d , 'r') as txtFile:
            for line in txtFile:
                if re.match(Class_i , line):
                    #print (line)
                    class_id = extract_class_id(line)
        #_________________________________________ audio preparation      
        
        audio_buf = read_audio_from_filename(wav_filename, target_sr=TARGET_SR)
        # normalize mean 0, variance 1
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        #print(i, wav_filename, original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            #print('PAD New length =', len(audio_buf))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
            #print('CUT New length =', len(audio_buf))

        #______________________________Split and save the data
        output_folder = OUTPUT_DIR_TRAIN
        if i % 50 == 0:
            output_folder = OUTPUT_DIR_VAL

        output_filename = os.path.join(output_folder, str(i) + '.pkl')

        out = {'class_id': class_id,
               'audio': audio_buf,
               'sr': TARGET_SR}
        with open(output_filename, 'wb') as w:
            pickle.dump(out, w)

#_______________________________________________Prepare test data
def test_dataloader(recordings):
    del_folder(OUTPUT_DIR_TEST)
    mkdir_p(OUTPUT_DIR_TEST)
    count = 0
    for i in recordings:
        audio_buf = i
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
        
        output_folder = OUTPUT_DIR_TEST
        output_filename = os.path.join(output_folder, str(count) + '.pkl')
        count = count + 1
        out = {'audio': audio_buf,
               'sr': TARGET_SR}
        with open(output_filename, 'wb') as w:
            pickle.dump(out, w)  


# if __name__ == '__main__':
    
#     convert_data()

