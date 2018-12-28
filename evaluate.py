import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

from dataset import get_video, get_csv

def moving_average(input, average):
    output = np.convolve(input, np.ones(average)/float(average), 'same')
    return output

def get_gt(csv_file):
    df = pd.read_csv(csv_file)
    gt_id = list(df['Id'])
    gt_f = list(df['F'])
    return gt_id, gt_f

class EvaluateModel():

    def __init__(self, video_file, csv_file, input_shape, model, lstm):
        self.video_file = video_file
        self.frame_count = get_video(self.video_file)['frame_count']
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.model = model
        self.steps = []
        if lstm == True:
            self.predict = self.output_lstm
        else:
            self.predict = self.output

    def output(self, frame):
        img = Image.fromarray(np.uint8(frame[:,:,:])).resize((self.input_shape[0], self.input_shape[1]))
        img = np.asarray(img)
        img = img[np.newaxis, :, :, :]
        img = img / 255.
        pred = self.model.predict(img)[0][0]
        return pred

    def output_lstm(self, frame):
        img = Image.fromarray(np.uint8(frame[:,:,:])).resize((self.input_shape[1], self.input_shape[2]))
        img = np.asarray(img)
        img = img / 255.
        self.steps.append(img)
        if len(self.steps) == self.input_shape[0]:
            inputs = np.array(self.steps)
            inputs = inputs[np.newaxis,...]
            pred = self.model.predict(inputs)[0][-1][0]
            self.steps  = self.steps[1:]
            return pred

    def compute_error(self, start_frame, end_frame):
        frames, preds, errors, answers = [], [], [], []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gt_id, self.gt_f = get_gt(self.csv_file)
        cap = cv2.VideoCapture(self.video_file)
        cap.set(1, start_frame)
        while(cap.isOpened()):
            flag, frame = cap.read()
            frame_id = cap.get(1)
            if flag == False or frame_id > end_frame:
                break
            if not frame_id in self.gt_id:
                continue
            pred = self.predict(frame)
            if not pred == None:
                frames.append(frame_id)
                preds.append(pred)
                answers.append(self.gt_f[self.gt_id.index(frame_id)])
                errors.append(np.absolute(self.gt_f[self.gt_id.index(frame_id)] - pred))
            sys.stdout.write("\r%d%s%d" % (frame_id, ' / ', end_frame))
            sys.stdout.flush()
        self.frames = np.array(frames)
        self.preds = np.array(preds)
        self.errors = np.array(errors)
        self.answers = np.array(answers)

    def show_result(self, average):
        gt_f = moving_average(self.answers, average)
        preds = moving_average(self.preds, average)
        sns.set()
        plt.figure(figsize=(16, 8))
        plt.plot(self.frames, preds, label='Prediction')
        plt.plot(self.frames, gt_f, label='GroundTruth')
        plt.legend()
        plt.show()