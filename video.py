
import os
import sys
import cv2
import shutil
import numpy as np
from PIL import Image

from src.fe_dataset import get_video, get_csv

def to_frames(video_file, image_dir, image_file='%s.jpg'):
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

def predict(video_file, csv_file, start_frame, model):
    video = get_video(video_file)
    cap = video['cap']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    input_shape = model.input_shape
    max_value = int(np.max(get_csv(csv_file)['F']))
    width, height = video['frame_width'], video['frame_height']
    xmin, xmax = int(width * 0.85), int(width * 0.90)
    ymax, ydif = int(height * 0.80), int(height * 0.05)
    cap.set(1, start_frame)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_id = cap.get(1)
        img = Image.fromarray(np.uint8(frame[:,:,:])).resize((input_shape[0], input_shape[1]))
        img = np.asarray(img)
        img = img[np.newaxis, :, :, :]
        pred = model.predict(img / 255.0)[0][0]
        meter = pred / max_value
        
        if pred > 3:
            color = (0,0,255)
        elif pred > 1.5:
            color = (0,255,0)
        else:
            color = (255,0,0)
            
        dst = cv2.rectangle(
            frame,
            (xmin, ymax + int(height * (-0.60 * meter))),
            (xmax, ymax), color, -1)
        
        dst = cv2.putText(
            dst, str(np.round(pred, 3)),
            (xmin, int(height * (-0.60 * meter + 0.78))), font, font_size, color, 2, cv2.LINE_AA)
        
        dst = cv2.putText(
            dst, str(int(frame_id)),
            (xmin, ymax + ydif), font, font_size, 'white', 2, cv2.LINE_AA)
        
        cv2.imshow('frame', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class VideoProcess:

    def __init__(self, video_file, target_dir, duration_second):
        self.video_file = video_file
        self.target_dir = target_dir
        self.video = cv2.VideoCapture(video_file)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_rate = 30
        self.duration = int(duration_second * self.frame_rate)

    # 目的のフレーム番号から指定した秒数だけ抜き出して保存する
    def extract(self, target_frame):
        self.video.set(1, target_frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        video_writer = cv2.VideoWriter(
            self.target_dir + self.video_file.replace('.mp4', '').split('/')[-1] + '_' + str(target_frame) + '.mp4',
            self.fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height))
        for _ in range(self.duration):
            is_capturing, frame = self.video.read()
            frame_num = self.video.get(1)
            if is_capturing:
                
                img = Image.fromarray(np.uint8(frame[:,:,:])).resize((input_shape[0], input_shape[1]))
                img = np.asarray(img)
                img = img[np.newaxis, :, :, :]
                pred = model.predict(img / 255.0)[0][0]
                max_force = np.max(force)
                force_meter = pred / max_force

                if pred > 3:
                    color = (0,0,255)
                elif pred > 1.5:
                    color = (0,255,0)
                else:
                    color = (255,0,0)
                dst = cv2.rectangle(frame,
                                    (int(frame.shape[1]*0.85), int(frame.shape[0]*(-0.6*force_meter+0.8))),
                                    (int(frame.shape[1]*0.9), int(frame.shape[0]*0.8)),
                                    color, -1)
                dst = cv2.putText(dst, str(np.round(pred, 3)),
                                  (int(frame.shape[1]*0.85), int(frame.shape[0]*(-0.6*force_meter+0.78))),
                                  font, 1, color, 2, cv2.LINE_AA)
                dst = cv2.putText(dst, str(int(frame_num)),
                                  (int(frame.shape[1]*0.85), int(frame.shape[0]*0.85)),
                                  font, 1, (255,255,255), 2, cv2.LINE_AA)
                
                video_writer.write(dst)
                sys.stdout.write("\r%d%s%d" % (frame_num, ' / ', max_frame))
                sys.stdout.flush()
            else:
                print('the end of video')