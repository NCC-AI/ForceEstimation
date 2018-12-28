import os
import sys
import cv2
import shutil
import numpy as np
import pandas as pd


class PathHolder():

    def __init__(self):
        self.project = '/home/takano/ForceEstimation/'

        self.data_raw = os.path.join(self.project, 'data/raw/')
        self.raw_video = os.path.join(self.data_raw, 'video.mp4')
        self.raw_csv = os.path.join(self.data_raw, 'forceinfo.csv')

        self.data_processed = os.path.join(self.project, 'data/processed/')
        self.sparse_csv = os.path.join(self.data_processed, 'sparse.csv')
        self.dense_csv = os.path.join(self.data_processed, 'dense.csv')
        self.frames = os.path.join(self.data_processed, 'frames/')

        self.models = os.path.join(self.project, 'models/')
        self.features = os.path.join(self.models, 'features/')
        self.targets = os.path.join(self.models, 'labels/')
        self.weights = os.path.join(self.models, 'weights/')


def get_video(video_file):
    output = {}
    output['cap'] = cv2.VideoCapture(video_file)
    output['fps'] = float(output['cap'].get(cv2.CAP_PROP_FPS))
    output['frame_count'] = int(output['cap'].get(cv2.CAP_PROP_FRAME_COUNT))
    output['frame_height'] = int(output['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    output['frame_width'] = int(output['cap'].get(cv2.CAP_PROP_FRAME_WIDTH))
    return output


def get_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


class CsvProcess():

    def __init__(self):
        self.path = PathHolder()
        self.truncation = -7 # Last 7 values are out of range for video.

    def copy_csv(self, raw, new):
        shutil.copy(raw, new)

    def convert_time(self, df, fps):
        # csvの行データをvideoのフレームに割り当てる
        times = []
        # 時間カラムを取得
        values = df['Time'].as_matrix()
        # Hourは統一されているので、Minutes+Seconds情報を取得
        for i in range(values.shape[0]):
            elements = values[i].split(':')
            minutes = elements[1:2]
            minutes.extend(elements[-1].split('.'))
            times.append(list(map(float, minutes)))
        # Minutes+Seconds情報をSeconds情報に変換
        times = np.array(times)
        seconds = times[:, 0] * 60 + times[:, 1] + (times[:, 2] / 100000)
        # 開始からの経過秒数に変換
        fromzero = seconds[:] - seconds[:][0]
        # frame_idとframe_name(path to .jpg)に変換
        frame_id = list(map(int, np.round(fromzero * fps, 1)))
        frame_name = [os.path.join(self.path.frames, '{0:06d}'.format(x)+'.jpg') for x in frame_id]
        return frame_id, frame_name

    def get_pf(self, df):
        # x, y, zデータを計算
        targets = df.drop(['Time'], axis=1).as_matrix()
        px, py, pz = np.absolute(targets[:, 0]), np.absolute(targets[:, 1]), np.absolute(targets[:, 2])
        fx, fy, fz = np.absolute(targets[:, 3]), np.absolute(targets[:, 4]), np.absolute(targets[:, 5])
        P = np.round(np.sqrt(np.power(px, 2) + np.power(py, 2) + np.power(pz, 2)), 3)
        F = np.round(np.sqrt(np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2)), 3)
        return P, F

    def update_csv(self):
        # 新たにcsvを作成して読み込み
        self.copy_csv(self.path.raw_csv, self.path.sparse_csv)
        df = pd.read_csv(self.path.sparse_csv)
        # 情報を取得
        fps = get_video(self.path.raw_video)['fps']
        frame_id, frame_name = self.convert_time(df, fps)
        P, F = self.get_pf(df)
        # csvの情報を更新してカラムを並び替え
        df['Time'], df['Id'], df['P'], df['F'] = frame_name, frame_id, P, F
        df = df.rename(columns={'Time': 'Path'})
        df = df.ix[:, ['Id', 'Path', 'Px','Py', 'Pz', 'P', 'Fx', 'Fy', 'Fz', 'F']]
        # 末端を切り落として上書き保存
        df[:self.truncation].to_csv(self.path.sparse_csv, index=False)

    def make_densecsv(self):
        # csvに存在しないフレーム情報を架空で作成する
        count = 0
        # 新たにcsvを作成して読み込み
        self.copy_csv(self.path.sparse_csv, self.path.dense_csv)
        frame_count = get_video(self.path.raw_video)['frame_count']
        data = np.array(pd.read_csv(self.path.dense_csv))
        # n行とn+1行を比較して、frame_idがスキップしていたら間を補填する
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
                cf, nf = data[i, -1], data[i+1, -1]
                new_id = cid + 1
                new_path = os.path.join(self.path.raw_frames, '{0:06d}'.format(new_id)+'.jpg')
                new_f = cf + (nf - cf)/(nid - cid)
                data = np.insert(data, i+1, data[i], axis=0)
                data[i+1][0] = new_id
                data[i+1][1] = new_path
                data[i+1][-1] = new_f
            count += 1
        # 並び替えて上書き保存
        dataframe = pd.DataFrame(data)
        dataframe.columns = ['Id', 'Path', 'Px', 'Py', 'Pz', 'P', 'Fx', 'Fy', 'Fz', 'F']
        dataframe.to_csv(self.path.dense_csv, index=False)