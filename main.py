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

sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import json
import models
import warnings
import xlsxwriter
import numpy as np
from tqdm import tqdm
from models import Model
from MBSTOI import mbstoi
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
from data_generator import DataGenerator
from keras.utils.layer_utils import count_params
from utils import setup, write_to_audio, gcc_phat, pad_and_separate


def train(args):
    if args.metric == "SNR":
        loss = models.SNR()
        args.output_activation = "tanh"
    else:
        args.metric == "SI-SDR"
        loss = models.SISDR()
        args.output_activation = "linear"

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    model_dir = os.path.join(args.model_dir, (args.topology + '_' + 'model[{}]').format(time_stamp))

    train_ds = DataGenerator("train", args).fetch()

    valid_ds = DataGenerator("valid", args).fetch()

    opt = tf.optimizers.Adam()

    os.makedirs(model_dir)

    args.mode = "test"

    json.dump(vars(args), open(os.path.join(model_dir, 'params.json'), 'w'), indent=4)

    log_path = os.path.join(model_dir, 'train_log[{}].csv'.format(time_stamp))

    checkpoint_path = os.path.join(model_dir, 'model_best[{}].h5'.format(time_stamp))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.8, patience=3, min_lr=0.),
        tf.keras.callbacks.CSVLogger(log_path),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           mode='auto', save_freq='epoch')]

    print("Training with " + args.metric + " loss.\n")

    model = Model(args).call()

    model.summary()

    model.compile(optimizer=opt, loss=loss)

    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=args.max_epoch, callbacks=callbacks)

    plt.style.use('ggplot')

    history_dict = history.history

    loss = [-_ for _ in history_dict['loss']]
    val_loss = [-_ for _ in history_dict['val_loss']]

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))
    plt.plot(epochs, loss, marker='.', label='Training')
    plt.plot(epochs, val_loss, marker='.', label='Validation')
    plt.title('Training and validation performance')
    plt.xlabel('Epoch')
    plt.grid(axis='x')
    plt.ylabel(args.metric)
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(model_dir, 'learning_curves.pdf'))


def evaluate(args, model):
    model_time_stamp = model[-20:-1]

    model_dir = os.path.join(args.model_dir, model)

    params_file = os.path.join(model_dir, "params.json")

    with open(params_file) as f:
        args = json.load(f)

    args = namedtuple("args", args.keys())(*args.values())

    if args.metric == "SNR":
        loss = models.SNR()
    else:
        args.metric == "SI-SDR"
        loss = models.SISDR()

    valid_ds = DataGenerator("valid", args).fetch()
    test_ds = DataGenerator("test", args).fetch()

    model = Model(args).call()

    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))

    model.compile(loss=loss)

    print("Evaluating model... \n")

    log_path_valid = os.path.join(model_dir, 'valid_log[{}].csv'.format(model_time_stamp))
    valid_logger = tf.keras.callbacks.CSVLogger(log_path_valid, separator=',', append=False)
    valid_logger.on_test_begin = valid_logger.on_train_begin
    valid_logger.on_test_batch_end = valid_logger.on_epoch_end
    valid_logger.on_test_end = valid_logger.on_train_end

    _, snr_left, snr_right = model.evaluate(valid_ds, callbacks=valid_logger, verbose=0)
    snr_valid_left = -snr_left
    snr_valid_right = -snr_right
    valid_snr = np.mean([snr_valid_left, snr_valid_right])

    print("Testing model... \n")

    log_path_test = os.path.join(model_dir, 'test_log[{}].csv'.format(model_time_stamp))
    test_logger = tf.keras.callbacks.CSVLogger(log_path_test, separator=',', append=False)
    test_logger.on_test_begin = test_logger.on_train_begin
    test_logger.on_test_batch_end = test_logger.on_epoch_end
    test_logger.on_test_end = test_logger.on_train_end

    _, snr_left, snr_right = model.evaluate(test_ds, callbacks=test_logger, verbose=0)
    snr_test_left = -snr_left
    snr_test_right = -snr_right
    test_snr = np.mean([snr_test_left, snr_test_right])

    print("Mean validation " + args.metric + ": {:.2f} dB".format(valid_snr))
    print("\nMean test " + args.metric + ": {:.2f} dB\n".format(test_snr))
    print("+++++++++++++++++++++++++++++++++++++++++\n")

    with open(os.path.join(model_dir, "Evaluation results[" + model_time_stamp + "].txt"), "w") as f:
        f.write("Topology: {:s}\n".format(args.topology))
        f.write("Mean test " + args.metric + ": {:.2f} dB\n".format(test_snr))
        f.write("Left test " + args.metric + ": {:.2f} dB\n".format(snr_test_left))
        f.write("Right test " + args.metric + ": {:.2f} dB\n".format(snr_test_right))
        f.write("Mean valid " + args.metric + ": {:.2f} dB\n".format(valid_snr))
        f.write("Left valid " + args.metric + ": {:.2f} dB\n".format(snr_valid_left))
        f.write("Right valid " + args.metric + ": {:.2f} dB\n".format(snr_valid_right))
        f.write("Number of parameters: {:d}\n".format(count_params(model.trainable_variables)))
        f.write("Attention size 1: {:d}\n".format(args.S))
        f.write("Attention size 2: {:d}\n".format(args.N))
        f.write("Loss function: {:s}\n".format(args.metric))
        f.write("Encoding: {:s}".format(args.encoding))

    f.close()

    workbook = xlsxwriter.Workbook(os.path.join(model_dir, "Evaluation results[" + model_time_stamp + "].xlsx"))
    worksheet = workbook.add_worksheet()
    data = (
        ['Topology', args.topology],
        ['Mean_test_' + args.metric, test_snr],
        ['Left_test_' + args.metric, snr_test_left],
        ['Right_test_' + args.metric, snr_test_right],
        ['Mean_valid_' + args.metric, valid_snr],
        ['Left_valid_' + args.metric, snr_valid_left],
        ['Right_valid_' + args.metric, snr_valid_right],
        ['Params', count_params(model.trainable_variables)],
        ['AttSize1', args.S],
        ['AttSize2', args.N],
        ['LossFunc', args.metric],
        ['Encoding', args.encoding])

    row = 0
    col = 0

    for var, value in data:
        worksheet.write(row, col, var)
        worksheet.write(row, col + 1, value)
        row += 1

    workbook.close()


def test(args, model):
    model_time_stamp = model[-20:-1]

    model_dir = os.path.join(args.model_dir, model)

    params_file = os.path.join(model_dir, "params.json")

    with open(params_file) as f:
        args = json.load(f)

    args = namedtuple("args", args.keys())(*args.values())

    if args.metric == "SNR":
        loss = models.SNR()
    else:
        args.metric == "SI-SDR"
        loss = models.SISDR()

    test_ds = DataGenerator("objective_test", args).fetch()

    model = Model(args).call()

    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))

    model.compile(loss=loss)

    model.summary()

    c = 2220

    print("\nComputing binaural cue errors...\n")
    i = 0
    itd_error = np.zeros(c)
    ild_error = np.zeros(c)
    snri = np.zeros(c)
    mbstoi_score = np.zeros(c)
    for inp, out in tqdm(test_ds, total=c, position=0, leave=True):
        p_left, p_right = pad_and_separate(inp, args, model)
        true_itd, _ = gcc_phat(out[0].numpy()[0], out[1].numpy()[0], args.sample_rate)
        est_itd, _ = gcc_phat(p_left.numpy()[0], p_right.numpy()[0], args.sample_rate)
        itd_error[i] = np.abs(est_itd) - np.abs(true_itd)

        true_ild = np.abs(10 * np.log10(np.mean(np.inner(out[0].numpy()[0], out[0].numpy()[0]))
                                        / np.mean(np.inner(out[1].numpy()[0], out[1].numpy()[0]))))
        est_ild = np.abs(10 * np.log10(np.mean(np.inner(p_left.numpy()[0], p_left.numpy()[0]))
                                       / np.mean(np.inner(p_right.numpy()[0], p_right.numpy()[0]))))
        ild_error[i] = np.abs(est_ild) - np.abs(true_ild)

        snr_l = 10 * np.log10(np.mean(out[0].numpy()[0] ** 2) / np.mean((out[0].numpy()[0] - p_left.numpy()[0]) ** 2))
        snr_r = 10 * np.log10(np.mean(out[1].numpy()[0] ** 2) / np.mean((out[1].numpy()[0] - p_right.numpy()[0]) ** 2))
        snr_pred = np.mean([snr_l, snr_r])

        snr_l = 10 * np.log10(np.mean(out[0].numpy()[0] ** 2) / np.mean((out[0].numpy()[0] - inp[0].numpy()[0]) ** 2))
        snr_r = 10 * np.log10(np.mean(out[1].numpy()[0] ** 2) / np.mean((out[1].numpy()[0] - inp[1].numpy()[0]) ** 2))
        snr_true = np.mean([snr_l, snr_r])

        snri[i] = snr_pred - snr_true

        mbstoi_score[i] = mbstoi(out[0].numpy()[0], out[1].numpy()[0], p_left.numpy()[0], p_right.numpy()[0])
        i += 1

    workbook = xlsxwriter.Workbook(os.path.join(model_dir, "Objective results[" + model_time_stamp + "].xlsx"))
    worksheet = workbook.add_worksheet()
    data = (
        ['ITD_ERROR'],
        ['ILD_ERROR'],
        ['SNRi'],
        ['MBSTOI'])

    col = 0

    for var in data:
        worksheet.write(0, col, var[0])
        col += 1

    for i in range(len(snri)):
        worksheet.write(i + 1, 0, itd_error[i])
        worksheet.write(i + 1, 1, ild_error[i])
        worksheet.write(i + 1, 2, snri[i])
        worksheet.write(i, 3, mbstoi_score[i])
    workbook.close()


def main() -> None:
    args = setup()
  
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print("\nUsing GPU... \n")
        args.GPU = True

    else:
        print('\nUsing CPU... \n')
        warnings.warn("Depth-wise convolution not available when using CPU for processing.")
        args.GPU = False

    if args.mode == "train":
        train(args)

    elif args.mode == "valid":
        trained_models = os.listdir(args.model_dir)
        for tm in trained_models:
            evaluate(args, tm)
    else:
        trained_models = os.listdir(args.model_dir)
        for tm in trained_models:
            test(args, tm)


if __name__ == '__main__':
    main()
