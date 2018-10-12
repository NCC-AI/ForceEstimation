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
from force_utils import DataProcessing

video_file = './dataset/video.mp4'
csv_file = './dataset/forceinfo.csv'
dataset = DataProcessing(video_file, csv_file, p=True, f=True, name=True)

# Split train and validation
keys = list(dataset.f_dict.keys())
train_num = int(round(0.8 * len(keys)))
train_keys = keys[:train_num]
val_keys = keys[train_num:]
val_num = len(val_keys)

input_shape = (299, 299, 3)
batch_size = 32
epochs = 30
learning_rate = 0.01

def generate_from_directory(train=True, batch_size=32):
    while True:
        x, y, i = [], [], 0
        if train:
            #random.shuffle(train_keys)
            for image_file in train_keys:
                image = load_img('dataset/frames/' + image_file, target_size=(input_shape[0], input_shape[1]))
                image = img_to_array(image) 
                image /= 255.
                x.append(image)
                y.append(dataset.f_dict[image_file])
                i += 1
                if i == batch_size:
                    yield (np.array(x), np.array(y))
                    x, y, i = [], [], 0
        else:
            #random.shuffle(val_keys)
            for image_file in val_keys:
                image = load_img('dataset/frames/' + image_file, target_size=(input_shape[0], input_shape[1]))
                image = img_to_array(image) 
                image /= 255.
                x.append(image)
                y.append(dataset.f_dict[image_file])
                i += 1
                if i == batch_size:
                    yield (np.array(x), np.array(y))
                    x, y, i = [], [], 0

base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))

bottleneck_features_train = base_model.predict_generator(
    generate_from_directory(True, 32),
    steps=train_num//batch_size,
    verbose=1)

np.save('dataset/features/dense_train.npy', bottleneck_features_train)

bottleneck_features_validation = base_model.predict_generator(
    generate_from_directory(False, 32),
    steps=val_num//batch_size,
    verbose=1)

np.save('dataset/features/dense_val.npy', bottleneck_features_validation)