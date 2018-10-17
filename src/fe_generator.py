
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array

def extract_generator(frames, mode_dict, train, input_shape, batch_size):
    while True:
        inputs, count = [], 0
        start, end = mode_dict[train]
        for i in range(start, end):
            image = load_img(frames[i], target_size=(input_shape[0], input_shape[1]))
            image = img_to_array(image) / 255.
            inputs.append(image)
            count += 1
            if count == batch_size:
                yield (np.array(inputs))
                inputs, count = [], 0