import data_loading
from data_loading import *
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd

data_ = data_loading.load_raw_dataset('ecg_data_200.json')          #dictionary type임, len = 2(x,y), x: 200,5000,12 y:200,5000,4

x_data = data_['x']
y_data = data_['y']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)


def DeleteandAugment(x_train, y_train):                                         # 0~2초, 8~10초 데이터는 사용하지 않음, 2~8초 사이 데이터를 증강(4초씩)
    aug_x_train = np.zeros((len(x_train)*3, 2000, len(x_train[0][0])))          # 480,2000,12
    aug_y_train = np.zeros((len(x_train)*3, 2000, len(y_train[0][0])))          # 480,2000,4

    for i, label in enumerate(y_train):  # (5000,12)
        aug_x_train[i*3] = x_train[i][1000:3000]     # 2~6초
        aug_x_train[i*3+1] = x_train[i][1500:3500]   # 3~7초
        aug_x_train[i*3+2] = x_train[i][2000:4000]   # 4~8초

        aug_y_train[i*3] = y_train[i][1000:3000]    # 2~6
        aug_y_train[i*3+1] = y_train[i][1500:3500]  # 3~7
        aug_y_train[i*3+2] = y_train[i][2000:4000]  # 4~8

    return aug_x_train, aug_y_train

