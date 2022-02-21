# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import os
import sys
import glob
import librosa
from tqdm import tqdm
import tensorflow as tf

class DataGenerator():
    def __init__(self, mode, args):
        if mode != "train" and mode != "valid" and mode != "test":
            raise ValueError("mode: {} while mode should be "
                             "'train', or 'test'".format(mode))
        
        if not os.path.isdir(args.data_dir):
            raise ValueError("cannot find data_dir: {}".format(args.data_dir))

        self.wav_dir = os.path.join(args.data_dir, mode)
        self.tfr = os.path.join(args.data_dir, mode + '.tfr')
        self.mode = mode
        self.batch_size = args.batch_size
        self.sample_rate = args.sample_rate
        self.duration = args.duration

        if not os.path.isfile(self.tfr):
            self._encode(self.mode)

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
            
    def fetch(self):
        dataset = tf.data.TFRecordDataset(self.tfr).map(self._decode,
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.mode == "train":
            dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
            train_dataset = dataset.batch(self.batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return train_dataset
        
        if self.mode == "valid":
            valid_dataset = dataset.batch(1, drop_remainder=True)
            valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE) 
            return valid_dataset
        
        else:
            dataset = dataset.batch(1, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

    def _encode(self, mode):
        
        if self.mode == "train":
            print("\nSerializing training data...\n")
        
        if self.mode == "valid":
            print("\nSerializing validation data...\n")
        
        if self.mode == "test":
            print("\nSerializing testing data...\n")
            
        writer = tf.io.TFRecordWriter(self.tfr)
        
        if self.mode != "test":
            mix_filenames = glob.glob(os.path.join(self.wav_dir, "*mixure.wav"))
            target_filenames = glob.glob(os.path.join(self.wav_dir, "*target.wav"))
            sys.stdout.flush()  
            
            for mix_filename, target_filename in tqdm(zip(mix_filenames, 
                                                          target_filenames), total = len(mix_filenames)):
                mix, _ = librosa.load(mix_filename, self.sample_rate, mono = False)
                clean, _ = librosa.load(target_filename, self.sample_rate, mono = False)
    
                def write(a, b):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "noisy_left" : self._float_list_feature(mix[0, a:b]),
                                "noisy_right": self._float_list_feature(mix[1, a:b]),
                                "clean_left" : self._float_list_feature(clean[0, a:b]),
                                "clean_right": self._float_list_feature(clean[1, a:b])}))
                    
                    writer.write(example.SerializeToString())
                
                now_length = mix.shape[-1]
                target_length = int(self.duration * self.sample_rate)
    
                if now_length < target_length:
                    continue 
                
                stride = int(self.duration * self.sample_rate)
                for i in range(0, now_length - target_length, stride):
                    write(i, i + target_length)
        else:
            mix_filenames = glob.glob(os.path.join(self.wav_dir, "*mixure.wav"))
            sys.stdout.flush()  
            
            for mix_filename in tqdm(mix_filenames, total = len(mix_filenames)):
                mix, _ = librosa.load(mix_filename, self.sample_rate, mono = False)
    
                def write(a, b):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "noisy_left" : self._float_list_feature(mix[0, a:b]),
                                "noisy_right": self._float_list_feature(mix[1, a:b])}))
                    
                    writer.write(example.SerializeToString())
                
                write(None, None)
        writer.close()
        sys.stdout.flush()
              
    def _decode(self, serialized_example):
        if self.mode != "test":
            example = tf.io.parse_single_example(
                serialized_example,
                features={
                    "noisy_left":  tf.io.VarLenFeature(tf.float32),
                    "noisy_right": tf.io.VarLenFeature(tf.float32),
                    "clean_left":  tf.io.VarLenFeature(tf.float32),
                    "clean_right": tf.io.VarLenFeature(tf.float32)})
            
            noisy_left =  tf.sparse.to_dense(example["noisy_left"])
     
            noisy_right = tf.sparse.to_dense(example["noisy_right"])
            
            clean_left =  tf.sparse.to_dense(example["clean_left"])
            
            clean_right = tf.sparse.to_dense(example["clean_right"])
            
            return (noisy_left, noisy_right), (clean_left, clean_right)
        else:
            example = tf.io.parse_single_example(
                serialized_example,
                features={
                    "noisy_left": tf.io.VarLenFeature(tf.float32),
                    "noisy_right": tf.io.VarLenFeature(tf.float32)})
            
            noisy_left =  tf.sparse.to_dense(example["noisy_left"])
     
            noisy_right = tf.sparse.to_dense(example["noisy_right"])
            
            return noisy_left, noisy_right
