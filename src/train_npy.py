# coding: utf-8

import sys
import os
import csv
import random
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from force_utils import database, save_history

# -------------------------------------------------------------------------------------
# GPU
# -------------------------------------------------------------------------------------

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0,1"
#set_session(tf.Session(config=config))

# -------------------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------------------

# Input
x_train = np.load('./dataset/features/bottleneck_features_train.npy')
x_test = np.load('./dataset/features/bottleneck_features_validation.npy')


# Make force_dic = {image_file_name : force_value}
movie_file = 'dataset/movie.mp4'
csv_file = 'dataset/forceinfo.csv'
dataset = database(movie_file, csv_file)
force = dataset.make_tension()
frame_id, frames = dataset.make_inputs()
force_dic = {}
for i in range(len(frames)):
    force_dic[frames[i]] = force[i]

# Split train and validation
keys = list(force_dic.keys())
values = list(force_dic.values())
train_num = int(round(0.8 * len(keys)))

# Target
y_train = values[:train_num]
for x in values[:30]:
    y_train.append(x)
y_test = values[train_num:]
for x in values[train_num:][:8]:
    y_test.append(x)

# -------------------------------------------------------------------------------------
# VARIABLE
# -------------------------------------------------------------------------------------

input_shape = (299, 299, 3)
batch_size = 32
epochs = 30
learning_rate = 0.01

# -------------------------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------------------------

base_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(1)(x)

model = Model(inputs=base_model.input, outputs=prediction)
for layer in base_model.layers:
        layer.trainable = False

# -------------------------------------------------------------------------------------
# COMPILE
# -------------------------------------------------------------------------------------

# Optimizer
optim = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Learning rate scheduler
def schedule(epoch, decay=0.9):
    return learning_rate * decay**(epoch)

# Callbacks
callbacks = [
    ModelCheckpoint('dataset/weights/weights_npy_{epoch:02d}.h5', monitor='val_loss', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')]
    # LearningRateScheduler(schedule)]

# Compile
model.compile(optimizer=optim, loss='mean_squared_error')

# -------------------------------------------------------------------------------------
# FIT
# -------------------------------------------------------------------------------------

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_test, y_test)
)

save_history(history, 'dataset/history/history_npy.txt')