
import os
import sys
import cv2
import shutil
import numpy as np
import pandas as pd

abspath    = '/home/takano/ForceEstimation/'
raw_video  = os.path.join(abspath, 'data/raw/video.mp4')
raw_csv    = os.path.join(abspath, 'data/raw/forceinfo.csv')
sparse_csv = os.path.join(abspath, 'data/processed/sparse.csv')
dense_csv  = os.path.join(abspath, 'data/processed/dense.csv')
raw_frames = os.path.join(abspath, 'data/processed/frames/')
truncation = -7 # Last 7 values are out of range for video.

def get_video(video_file):
    output = {}
    output['cap']          = cv2.VideoCapture(video_file)
    output['fps']          = float(output['cap'].get(cv2.CAP_PROP_FPS))
    output['frame_count']  = int(output['cap'].get(cv2.CAP_PROP_FRAME_COUNT))
    output['frame_height'] = int(output['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    output['frame_width']  = int(output['cap'].get(cv2.CAP_PROP_FRAME_WIDTH))
    return output

def get_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def copy_csv(original, new):
    shutil.copy(original, new)

def convert_time(df, fps):
    times  = []
    values = df['Time'].as_matrix()
    for i in range(values.shape[0]):
        elements = values[i].split(':')
        minutes  = elements[1:2]
        minutes.extend(elements[-1].split('.'))
        times.append(list(map(float, minutes)))
    times      = np.array(times)
    seconds    = times[:, 0] * 60 + times[:, 1] + (times[:, 2] / 100000)
    fromzero   = seconds[:] - seconds[:][0]
    frame_id   = list(map(int, np.round(fromzero * fps, 1)))
    frame_name = [os.path.join(raw_frames, '{0:06d}'.format(x)+'.jpg') for x in frame_id]
    return frame_id, frame_name

def get_pf(df):
    targets    = df.drop(['Time'], axis=1).as_matrix()
    px, py, pz = np.absolute(targets[:, 0]), np.absolute(targets[:, 1]), np.absolute(targets[:, 2])
    fx, fy, fz = np.absolute(targets[:, 3]), np.absolute(targets[:, 4]), np.absolute(targets[:, 5])
    P = np.round(np.sqrt(np.power(px, 2) + np.power(py, 2) + np.power(pz, 2)), 3)
    F = np.round(np.sqrt(np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2)), 3)
    return P, F

def update_sparsecsv():
    copy_csv(raw_csv, sparse_csv)
    df = pd.read_csv(sparse_csv)
    fps = get_video(raw_video)['fps']
    frame_id, frame_name = convert_time(df, fps)
    P, F = get_pf(df)
    df['Time'], df['Id'], df['P'], df['F'] = frame_name, frame_id, P, F
    df = df.rename(columns={'Time': 'Path'})
    df = df.ix[:, ['Id', 'Path', 'Px','Py', 'Pz', 'P', 'Fx', 'Fy', 'Fz', 'F']]
    df[:truncation].to_csv(sparse_csv, index=False)

def make_densecsv():
    count = 0
    copy_csv(sparse_csv, dense_csv)
    frame_count = get_video(raw_video)['frame_count']
    data = np.array(pd.read_csv(dense_csv))
    for i in np.zeros(frame_count*2, dtype='int'):
        i += count
        if i == frame_count-1:
            break
        sys.stdout.write("\r%d%s%d" % (i, ' / ', frame_count))
        sys.stdout.flush()
        cid, nid = data[i, 0], data[i+1, 0]
        if cid == nid:
            data = np.delete(data, i+1, axis=0)
            continue
        if nid - cid > 1:
            cf, nf        = data[i, -1], data[i+1, -1]
            new_id        = cid + 1
            new_path      = os.path.join(raw_frames, '{0:06d}'.format(new_id)+'.jpg')
            new_f         = cf + (nf - cf)/(nid - cid)
            data          = np.insert(data, i+1, data[i], axis=0)
            data[i+1][0]  = new_id
            data[i+1][1]  = new_path
            data[i+1][-1] = new_f
        count += 1
    dataframe = pd.DataFrame(data)
    dataframe.columns = ['Id', 'Path', 'Px', 'Py', 'Pz', 'P', 'Fx', 'Fy', 'Fz', 'F']
    dataframe.to_csv(dense_csv, index=False)