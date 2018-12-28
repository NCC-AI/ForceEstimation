import os
import sys
sys.path.append(os.pardir)
import argparse
import numpy as np

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import LSTM, TimeDistributed
from keras.applications import VGG16, VGG19, Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

from dataset import get_csv
from generator import ExtractGenerator


csv_file = '../data/processed/dense.csv'
inputs = get_csv(csv_file)['Path']
targets = get_csv(csv_file)['F']

x_train_path = '../models/features/x_train.npy'
y_train_path = '../models/features/y_train.npy'
x_test_path = '../models/features/x_test.npy'
y_test_path = '../models/features/y_test.npy'

timesteps = 5
height, width, channel = 299, 299, 3


def train_with_features():
    """ Train model using bottleneck features.
    """
    x_train, y_train = np.load(x_train_path), np.load(y_train_path)
    x_test, y_test = np.load(x_test_path), np.load(y_test_path)

    # Reshape training set
    sumple_train_num = len(x_train)//timesteps
    height, width, channel = x_train.shape[1:]
    x_train = x_train[:sumple_train_num].reshape(sumple_train_num/timesteps, timesteps, height, width, channel)
    y_train = y_train[:sumple_train_num].reshape(sumple_train_num/timesteps, timesteps, 1)
    
    # Reshape validation set
    sumple_test_num = len(x_test)//timesteps
    height, width, channel = x_test.shape[1:]
    x_test = x_test[:sumple_test_num].reshape(sumple_test_num/timesteps, timesteps, height, width, channel)
    y_test = y_test[:sumple_test_num].reshape(sumple_test_num/timesteps, timesteps, height, width, channel)

    input_tensor = Input(shape=x_train.shape[1:])
    net = TimeDistributed(Flatten()(input_tensor))
    net = LSTM(512, return_sequences=True)(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(1)(net)
    model = Model(inputs=input_tensor, outputs=net)

    optimizer = keras.optimizers.RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'))
    checkpoint = '../models/recurrent_bottleneck_weights.h5'
    callbacks.append(ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True))

    model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=100,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_test, y_test)
    )


def train_with_generator():
    """ Train model with generator.
    """
