from __future__ import print_function
import sklearn.ensemble
from sklearn import metrics
from myo import init, Hub, DeviceListener, StreamEmg, Feed
from time import sleep
import numpy as np
import threading
import collections
import math
import pandas as pd

MAX_GESTURE = 10

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def rms(array):
    n = len(array)
    sum = 0
    for a in array:
        sum =+ a*a
    return np.sqrt((1/float(n))*sum)

def iav(array):
    sum = 0
    for a in array:
        sum += np.abs(a)
    return sum

def ssi(array):
    sum = 0
    for a in array:
        sum += a*a
    return sum

def var(array):
    n = len(array)
    sum = 0
    for a in array:
        sum += a*a
    return ((1/float(n-1))*sum)

def tm3(array):
    n = len(array)
    print('n : ', n)
    sum = 0
    for a in array:
        sum =+ a*a*a
    return np.power((1/float(n))*sum,1/float(3))

def wl(array):
    sum = 0
    for a in range(0,len(array)-1):
        sum =+ array[a+1] - array[a]
    return sum

def aac(array):
    n = len(array)
    sum = 0
    for a in range(0,n-1):
        sum =+ array[0+1] - array[0]
    return sum/float(n)


def featurize(array):
    n = []
    for a in array:
        n.append(rms(a))
    return n

status = 0

X = []

def toEuler(quat):
    quat = quat[0]

    # Roll
    sin = 2.0 * (quat.w * quat.w + quat.y * quat.z)
    cos = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    roll = math.atan2(sin, cos)

    # Pitch
    pitch = math.asin(2 * (quat.w * quat.y - quat.z * quat.x))

    # Yaw
    sin = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cos = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    yaw = math.atan2(sin, cos)
    return [pitch, roll, yaw]

class Listener(DeviceListener):
    def __init__(self, queue_size=1):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_connect(self, myo, timestamp, firmware_version):
        myo.set_stream_emg(StreamEmg.enabled)

    def on_emg_data(self, myo, timestamp, emg):
        if(status):
            X.append(np.asarray(emg))

    def on_orientation_data(self, myo, timestamp, quat):
        # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        with self.lock:
            self.ori_data_queue.append(quat)

    def get_ori_data(self):
        with self.lock:
            return list(self.ori_data_queue)


init('/Users/sosoon/Downloads/sdk/myo.framework')

hub = Hub()
feed = Feed()
listener = Listener()
hub.run(1000, listener)

status = 9999

sleep(1)

myX = []

req_iter = 20
train_n = []


ges1 = ['Pain in neck' , 'headache', 'Injection', 'Hearing-Aid', 'Nurse' , 'Blood Pressure', 'Surgery', 'Test', 'Prescription', 'Wheelchair']
ges2 = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
ges3 = ['Spread Fingers', 'Wave Out', 'Wave In', 'Fist', 'Rest']

ges = ges1

for a in range(1,4):

    for i in range(MAX_GESTURE):
        print("\nGesture -- ", ges[i], " : Ready?")
        input("Press Enter to continue...")
        X = []
        train_temp = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_temp.append(np.asarray(X))
                X = []
                if len(train_temp) > a*req_iter:
                    train_n.append(train_temp)
                    break

train_x = []
train_y = []

i = 1
for n in range(MAX_GESTURE):
    for a in train_n[n]:
        train_x.append(np.asarray(a))
        train_y.append(i)
    i += 1


train_x_f = []

for a in train_x:
    x_f_h = []
    for b in range(0,8):
        x_f_h.append(rms(a[:, b]))
        x_f_h.append(iav(a[:, b]))
        x_f_h.append(ssi(a[:, b]))
        x_f_h.append(var(a[:, b]))
        # x_f_h.append(tm3(a[:, b]))
        x_f_h.append(wl(a[:, b]))
        x_f_h.append(aac(a[:, b]))
    train_x_f.append(x_f_h)

train_x_f_dataframe = pd.DataFrame(train_x_f)
train_y_dataframe = pd.DataFrame(train_y)

train_x_f_dataframe.to_csv("/Users/sosoon/coding/myo_train_data/train_x.csv",header=False, index=False)
train_y_dataframe.to_csv("/Users/sosoon/coding/myo_train_data/train_y.csv",header=False, index=False)
