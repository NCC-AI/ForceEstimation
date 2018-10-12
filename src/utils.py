import os
import sys
import shutil
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class VideoProcess():
    def __init__(self, video_file):
        self.cap          = cv2.VideoCapture(video_file)
        self.frame_count  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

class CsvProcess():
    def __init__(self, csv_file):
        self.truncation = -7
        self.df         = pd.read_csv(csv_file)
        #self.df_inputs  = self.df['Time'].as_matrix()[:self.truncation]
        #self.df_targets = self.df.drop(["Time"], axis=1).as_matrix()[:self.truncation]

def make_dataset():
    new_time = []
    truncation = -7
    dataset = pd.read_csv('dataset/forceinfo.csv').values
    for i in range(dataset.shape[0]):
        times = dataset[:, 0][i].split(':')
        minutes = time[1]
        seconds = time[-1].split('.')
        minutes.extend(seconds)
        new_time.append(list(map(float, minutes)))
    new_time = np.array(new_time)
    new_time = new_time[:, 0] * 60 + new_time[:, 1] + (new_time[:, 2] / 100000)
    time_fromzero = time[:] - time[:][0]
    frame_id = list(map(int, np.round(time_fromzero * self.fps, 1)))
    frame_name = ['{0:06d}'.format(x)+'.jpg' for x in frame_id]




    

    def make_inputs(self):
        time = []
        for i in range(self.df_time.shape[0]):
            time_elements = self.df_time[i].split(':')
            minutes = time_elements[1:2]
            seconds = time_elements[-1].split('.')
            minutes.extend(seconds)
            time.append(list(map(float, minutes)))
        time = np.array(time)
        time = time[:, 0] * 60 + time[:, 1] + (time[:, 2] / 100000)
        time_fromzero = time[:] - time[:][0]
        frame_id = list(map(int, np.round(time_fromzero * self.fps, 1)))
        frame_name = ['{0:06d}'.format(x)+'.jpg' for x in frame_id]

        return frame_id, frame_name

    def make_targets(self, p=True, f=True):
        px, py, pz = self.df_target[:, 0], self.df_target[:, 1], self.df_target[:, 2]
        fx, fy, fz = self.df_target[:, 3], self.df_target[:, 4], self.df_target[:, 5]
        p_target, f_target =  (px, py, pz), (fx, fy, fz)
        if p == True:
            px, py, pz = np.absolute(px), np.absolute(py), np.absolute(pz)
            p_target = np.round(np.sqrt(np.power(px, 2) + np.power(py, 2) + np.power(pz, 2)), 3)
        if f == True:
            fx, fy, fz = np.absolute(fx), np.absolute(fy), np.absolute(fz)
            f_target = np.round(np.sqrt(np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2)), 3)

        return p_target, f_target

    def make_pairs(self, name=False):
        p_dict, f_dict = {}, {}
        if name == False:
            for i in range(len(self.frame_name)):
                p_dict[self.frame_id[i]] = self.p_target[:][i]
                f_dict[self.frame_id[i]] = self.f_target[:][i]
        else:
            for i in range(len(self.frame_name)):
                p_dict[self.frame_name[i]] = self.p_target[i]
                f_dict[self.frame_name[i]] = self.f_target[i]

        return p_dict, f_dict

def evaluate_model(dataset, input_shape, f, lstm, timesteps, base_model, top_model, start_frame):
    if f == True: target = dataset.f_dict
    else: target = dataset.p_dict
    keys, values = list(target.keys()), list(target.values())
    features, frames, preds, errors = [], [], [], []

    dataset.cap.set(1, start_frame)
    while(dataset.cap.isOpened()):
        flag, frame = dataset.cap.read()
        frame_num = dataset.cap.get(1)
        if flag == False:
            break
        if not frame_num in keys:
            continue
        img = Image.fromarray(np.uint8(frame[:,:,:])).resize((input_shape[0], input_shape[1]))
        img = np.asarray(img)
        img = img[np.newaxis, :, :, :]
        frames.append(frame_num)
        base_output = base_model.predict(img / 255.0)
        if lstm == True:
            features.extend(base_output)
            if len(features) == timesteps:
                features_for_lstm = np.array(features)
                features = features[1:]
                features_for_lstm = features_for_lstm[np.newaxis,...]
                pred = top_model.predict(features_for_lstm)[0][0]
                preds.append(pred)
                errors.append(np.absolute(target[frame_num] - pred))
        else:
            pred = top_model.predict(base_output)[0][0]
            preds.append(pred)
            errors.append(np.absolute(target[frame_num] - pred))
        sys.stdout.write("\r%d%s%d" % (frame_num, ' / ', dataset.frame_count))
        sys.stdout.flush()

    return np.array(frames), np.array(preds), np.array(errors)

def video_to_frames(video_file, image_dir, image_file='%s.jpg'):
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    i = 0
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        flag, frame = cap.read()
        if flag == False:
            break
        cv2.imwrite(image_dir + image_file % str(i).zfill(6), frame)
        print('Save', image_dir + image_file % str(i).zfill(6))
        i += 1
    cap.release()