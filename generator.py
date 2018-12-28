import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array

from dataset import get_csv

class Generator():

    def __init__(self, csv_file, mode_dict, input_shape, batch_size):
        self.csv_file = csv_file
        self.mode_dict = mode_dict
        self.frames = np.array(get_csv(csv_file)['Path'])
        self.force = np.array(get_csv(csv_file)['F'])
        self.input_shape = input_shape
        self.batch_size = batch_size

    def generate(self, train):
        self.targets = []
        inputs = []
        count = 0
        start, end = self.mode_dict[train]
        target_size = (self.input_shape[0], self.input_shape[1])
        while True:
            for i in range(start, end):
                img = load_img(self.frames[i], target_size=target_size)
                img = img_to_array(img) / 255.
                inputs.append(img)
                self.targets.append(self.force[i])
                count += 1
                if count == self.batch_size:
                    yield np.array(inputs)
                    inputs, count = [], 0

    def series_generate(self, train):
        self.targets = []
        batch_in, times_in = [], []
        batch_ta, times_ta = [], []
        start, end = self.mode_dict[train]
        target_size = (self.input_shape[1], self.input_shape[2])
        while True:
            for i in range(start, end):
                img = load_img(self.frames[i], target_size=target_size)
                img = img_to_array(img) / 255.
                times_in.append(img)
                times_ta.append([self.force[i]])
                if len(times_in) == self.input_shape[0]:
                    batch_in.append(times_in)
                    batch_ta.append(times_ta)
                    times_in, times_ta = [], []
                if len(batch_in) == self.batch_size:
                    yield np.array(batch_in)
                    self.targets.extend(batch_ta)
                    batch_in, batch_ta = [], []

class ExtractGenerator():

    def __init__(self, inputs, targets, input_shape=(299, 299, 3), batch_size=32, shuffle=False):
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)        
        self.target_size = (input_shape[0], input_shape[1])
        self.batch_size = batch_size
        self.inputs_len = len(inputs)
        if shuffle == True:
            self.shuffle_data()
        self.dataset = self.split_data()

    def shuffle_data(self):
        p = np.random.permutation(self.inputs_len)
        self.inputs = self.inputs[p]
        self.targets = self.targets[p]

    def split_data(self):
        self.num_train = int(self.inputs_len * 0.8)
        self.num_val = self.inputs_len - self.num_train
        x_train = self.inputs[:self.num_train]
        y_train = self.targets[:self.num_train]
        x_test = self.inputs[self.num_train:]
        y_test = self.targets[self.num_train:]
        
        return {1:(x_train, y_train), 0:(x_test, y_test)}

    def generate(self, train):
        self.labels, steps = [], []
        images, count = [], 0
        x_data, y_data = self.dataset[train]
        while True:
            for i in range(len(x_data)):
                img = load_img(x_data[i], target_size=self.target_size)
                img = img_to_array(img) / 255.
                images.append(img)
                steps.append(y_data[i])
                count += 1
                if count == self.batch_size:
                    yield np.array(images)
                    self.labels.extend(steps)
                    images, steps, count = [], [], 0