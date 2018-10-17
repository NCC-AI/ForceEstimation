
import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

from fe_dataset import get_video, get_csv

def moving_average(input, average):
    output = np.convolve(input, np.ones(average)/float(average), 'same')
    return output

def get_gt(csv_file):
    df    = pd.read_csv(csv_file)
    gt_id = list(df['Id'])
    gt_f  = list(df['F'])
    return gt_id, gt_f

class EvaluateModel():

    def __init__(self, video_file, csv_file):
        self.video_file  = video_file
        self.frame_count = get_video(self.video_file)['frame_count']
        self.csv_file    = csv_file

    def compute_error(self, input_shape, start_frame, end_frame, model):
        frames, preds, errors, answers = [], [], [], []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gt_id, self.gt_f = get_gt(self.csv_file)
        cap = cv2.VideoCapture(self.video_file)
        cap.set(1, start_frame)
        while(cap.isOpened()):
            flag, frame = cap.read()
            frame_id    = cap.get(1)
            if flag == False or frame_id > end_frame:
                break
            if not frame_id in self.gt_id:
                continue
            frames.append(frame_id)
            img = Image.fromarray(np.uint8(frame[:,:,:])).resize((input_shape[0], input_shape[1]))
            img = np.asarray(img)
            img = img[np.newaxis, :, :, :]
            pred     = model.predict(img / 255.)[0][0]
            preds.append(pred)
            answers.append(self.gt_f[self.gt_id.index(frame_id)])
            errors.append(np.absolute(self.gt_f[self.gt_id.index(frame_id)] - pred))
            sys.stdout.write("\r%d%s%d" % (frame_id, ' / ', end_frame))
            sys.stdout.flush()
        self.frames  = np.array(frames)
        self.preds   = np.array(preds)
        self.errors  = np.array(errors)
        self.answers = np.array(answers)

    def show_result(self, average):
        # gt_f  = self.gt_f[self.start_frame+1:self.end_frame+1]
        # gt_f  = self.gt_f[self.start_frame:len(self.frames)+1]
        # gt_f  = moving_average(gt_f, average)
        gt_f = moving_average(self.answers, average)
        preds = moving_average(self.preds, average)
        sns.set()
        plt.figure(figsize=(16, 8))
        plt.plot(self.frames, preds, label='Prediction')
        plt.plot(self.frames, gt_f, label='GroundTruth')
        plt.legend()
        plt.show()