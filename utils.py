# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import resample

def setup():
    parser = argparse.ArgumentParser(description = 'Main configuration')
    
    parser.add_argument('-top', '--topology', type=str, default = "Double_attention")
    
    parser.add_argument('-mo', '--mode', type=str, default='test')  
    
    parser.add_argument('-gpu',  '--GPU', type=bool, default = True)
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-ld', '--model_dir', type=str, default='./models')    
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-md', '--metadata_dir', type=str, default='./data/test/metadata')
    parser.add_argument('-sr', '--sample_rate', type=int, default=8000)
    parser.add_argument('-c',  '--causal', type=bool, default = True)
    parser.add_argument('-me', '--max_epoch', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-k',  '--skip', type=bool, default = True)
    parser.add_argument('-d',  '--duration', type=int, default=4)
     
    parser.add_argument('-N', '-N', type=int, default=64)
    parser.add_argument('-L', '-L', type=int, default=16)
    parser.add_argument('-B', '-B', type=int, default=64)
    parser.add_argument('-H', '-H', type=int, default=64)
    parser.add_argument('-S', '-S', type=int, default=128)
    parser.add_argument('-P', '-P', type=int, default=128)
    parser.add_argument('-X', '-X', type=int, default=2)
    parser.add_argument('-R', '-R', type=int, default=2)
    parser.add_argument('-G', '-G', type=int, default=16)
     
    args = parser.parse_args()

    return args

def plot_spectrograms(model, ds, path, args, n_predictions): 
    if n_predictions > 10:
        n_predictions = 5
        
    elements = ds.take(n_predictions)
    
    noisy_l = np.zeros((n_predictions, args.duration*args.sample_rate))
    clean_l = np.zeros((n_predictions, args.duration*args.sample_rate))
    noisy_r = np.zeros((n_predictions, args.duration*args.sample_rate))
    clean_r = np.zeros((n_predictions, args.duration*args.sample_rate))

    i = 0
    for x, _ in elements:
        noisy_l[i,:] = x[0][0]
        clean_l[i,:] = model(x)[0][0]
        noisy_r[i,:] = x[1][0]
        clean_r[i,:] = model(x)[1][0]
        i +=1
        
    rows = n_predictions
    cols = 4
    
    plt.style.use('seaborn-poster')
    
    f, arr = plt.subplots(rows, cols, figsize=(10*cols, 4 * rows))
    arr[0,0].set_title('Noisy Left')
    arr[0,1].set_title('Enhanced Left')
    arr[0,2].set_title('Noisy Right')
    arr[0,3].set_title('Enhanced Right')
    arr[n_predictions-1,0].set_xlabel('Time (s)')
    arr[n_predictions-1,1].set_xlabel('Time (s)')
    arr[n_predictions-1,2].set_xlabel('Time (s)')
    arr[n_predictions-1,3].set_xlabel('Time (s)')
    
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                p = arr[r, c]
                p.set_ylabel("Frequency (Hz)")
                p.specgram(noisy_l[r,:], NFFT = 128, Fs = args.sample_rate, 
                           noverlap=64, cmap = "Blues")
                p.grid(False)
            elif c == 1:
                p = arr[r, c]
                p.specgram(clean_l[r,:], NFFT = 128, Fs = args.sample_rate, 
                           noverlap=64, cmap = 'Blues')
                p.grid(False)
            elif c == 2:
                p = arr[r, c]
                p.specgram(noisy_r[r,:], NFFT = 128, Fs = args.sample_rate, 
                           noverlap=64, cmap = 'Reds')
                p.grid(False)
            else:
                p = arr[r, c]
                p.specgram(clean_r[r,:], NFFT = 128, Fs = args.sample_rate, 
                           noverlap=64, cmap = 'Reds')
                p.grid(False)
    plt.savefig(path)

def write_to_audio(model, ds, args, path):   
    c = 0
    for inp in ds:
        c += 1
    j = 1
    i = 0
  
    print("\nWriting predictions to audio...\n")
    
    for inp in tqdm(ds, total = c, position = 0, leave = True): 

        prediction_left, prediction_right = pad_and_separate(inp, args, model)
        l = len(prediction_left[-1])
        
        prediction_left = resample(prediction_left[-1], int(l * (44100 / args.sample_rate) + 1))

        prediction_right = resample(prediction_right[-1], int(l * (44100 / args.sample_rate) + 1))
        
        predicted = tf.stack([prediction_left, prediction_right], 0)
        
        predicted = predicted/np.max(np.abs(predicted))
        
        fname = os.path.join(path, '{}_HA-output.wav'.format(i))
        string = tf.audio.encode_wav(tf.transpose(predicted), 44100)
        tf.io.write_file(fname, string, name='{}_HA-output.wav'.format(i))
        j += 1
        i += 1
        if j == c-1:
            print("\nDone!")
            
def pad_and_separate(inp, args, model, return_padded = False):

    original_length = inp[0].shape[-1]
    slices = (original_length -tf.math.floormod(original_length, 
                                                args.duration*args.sample_rate).numpy())//(args.duration*args.sample_rate) + 1
    left_inp_pad = np.zeros((1, slices*args.sample_rate*args.duration), dtype = np.float32)
    right_inp_pad = np.zeros((1, slices*args.sample_rate*args.duration), dtype = np.float32)
    left_inp_pad[0][0:original_length] = inp[0][0]
    right_inp_pad[0][0:original_length] = inp[1][0]
        
    for i in range(slices):
        prediction = model((left_inp_pad[:,i*args.sample_rate*args.duration:(i+1)*args.sample_rate*args.duration], 
                            right_inp_pad[:, i*args.sample_rate*args.duration:(i+1)*args.sample_rate*args.duration]))
        if i == 0:
            prediction_left = prediction[0]
            prediction_right = prediction[1]
        else:
            prediction_left = tf.concat([prediction_left, prediction[0]], axis = 1)
            prediction_right = tf.concat([prediction_right, prediction[1]], axis = 1)
    
    return prediction_left[:,0:original_length], prediction_right[:,0:original_length]