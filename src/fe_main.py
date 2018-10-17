
import argparse
import os, sys
import csv
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend.tensorflow_backend import set_session

from fe_dataset import get_csv
from fe_models import EXTRACTOR, REGRESSOR
from fe_generator import extract_generator

# ArgumentParser
parser = argparse.ArgumentParser(description='Simple script for extract features or train regressor.')
parser.add_argument('function', help='Function name, extract or regress.')
args   = parser.parse_args()

# Path
abspath     = '/home/takano/ForceEstimation/'
video_file  = os.path.join(abspath, 'data/raw/video.mp4')
csv_file    = os.path.join(abspath, 'data/processed/sparse.csv')
path_prefix = '_' + csv_file[csv_file.rfind('/')+1:csv_file.rfind('/')+3]
features    = os.path.join(abspath, 'data/interim/features/')
weights     = os.path.join(abspath, 'data/interim/weights/')

# Parameter
input_shape = (299, 299, 3)
batch_size  = 1
epochs      = 100
backbone    = 'VGG16'

# Dataset
frames      = np.array(get_csv(csv_file)['Path'])
num_train   = int(len(frames) * 0.8)
num_val     = len(frames) - num_train
mode_dict   = {1:(0, num_train), 0:(num_train, len(frames))}

# Model
extractor   = EXTRACTOR(input_shape=input_shape, backbone=backbone)
regressor   = REGRESSOR(extractor, weights=None)

# Extract
def extract():
    features_train = extractor.predict_generator(
        extract_generator(frames=frames, mode_dict=mode_dict, train=True, input_shape=input_shape, batch_size=batch_size),
        steps=num_train//batch_size,
        verbose=1)
    np.save(os.path.join(features, backbone + path_prefix + '_train.npy'), features_train)

    features_validation = extractor.predict_generator(
        extract_generator(frames=frames, mode_dict=mode_dict, train=False, input_shape=input_shape, batch_size=batch_size),
        steps=num_val//batch_size,
        verbose=1)
    np.save(os.path.join(features, backbone + path_prefix + '_val.npy'), features_validation)

# Regress
def regress():
    x_train   = np.load(os.path.join(features, backbone + path_prefix + '_train.npy'))
    x_test    = np.load(os.path.join(features, backbone + path_prefix + '_val.npy'))
    targets   = np.array(get_csv(csv_file)['F'])
    dif_train = len(targets[:num_train]) - len(x_train)
    dif_test  = len(targets[num_train:]) - len(x_test)
    y_train   = targets[:num_train-dif_train]
    y_test    = targets[num_train:-dif_test]

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    def outlier_loss(y_true, y_pred):
        return K.switch(K.mean(y_true) > 2.0, tf.multiply(K.mean(K.pow((y_pred - y_true), 2), axis=-1), 10), K.mean(K.pow((y_pred - y_true), 2), axis=-1))
        # return K.mean(K.pow((y_pred - y_true), 2), axis=-1)

    callbacks = [
                ModelCheckpoint(os.path.join(weights, backbone + path_prefix + '.h5'), monitor='val_loss', verbose=1, save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
                ]

    regressor.compile(optimizer=optimizer, loss=outlier_loss)
    regressor.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        # shuffle=True,
        callbacks=callbacks,
        verbose=1,
        validation_data=(x_test, y_test)
    )

if __name__ == '__main__':
    eval(args.function)()