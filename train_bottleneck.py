import os
import sys
sys.path.append(os.pardir)
import argparse
import numpy as np

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications import VGG16, VGG19, Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

from dataset import get_csv
from generator import ExtractGenerator


parser = argparse.ArgumentParser(description='Extract features or Train model with features.')
parser.add_argument('function', choices=['extract', 'train'], help='function name extract or train.')
args = parser.parse_args()

csv_file = '../data/processed/sparse.csv'
inputs = get_csv(csv_file)['Path']
targets = get_csv(csv_file)['F']

x_train_path = '../models/features/x_train.npy'
y_train_path = '../models/features/y_train.npy'
x_test_path = '../models/features/x_test.npy'
y_test_path = '../models/features/y_test.npy'

batch_size = 32
input_shape = (299, 299, 3)


def extract():
    """ Extract features using pretrained model.
    """
    model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    gen = ExtractGenerator(inputs, targets)

    # Training data
    features = model.predict_generator(
        gen.generate(train=True),
        steps=gen.num_train//batch_size,
        verbose=1
    )
    np.save(x_train_path, features)
    np.save(y_train_path, gen.labels)

    # Validation data
    features = model.predict_generator(
        gen.generate(train=False),
        steps=gen.num_val//batch_size,
        verbose=1
    )
    np.save(x_test_path, features)
    np.save(y_test_path, gen.labels)


def train():
    """ Train model using bottleneck features.
    """
    x_train, y_train = np.load(x_train_path), np.load(y_train_path)
    x_test, y_test = np.load(x_test_path), np.load(y_test_path)
    
    input_tensor = Input(shape=x_train.shape[1:])
    net = Flatten()(input_tensor)
    net = Dense(512, activation='relu')(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(1)(net)
    model = Model(inputs=input_tensor, outputs=net)

    optimizer = keras.optimizers.RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'))
    checkpoint = '../models/bottleneck_weights.h5'
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


if __name__ == '__main__':
    eval(args.function)()