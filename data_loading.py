"""
Code Source
https://github.com/Namenaro/ecg_segmentation
(Paper : ECG Segmentation by Neural Networks : Errors and Correction)

Read LUDB dataset (json file)
"""
import numpy as np
import json
from variables import *
import os

leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']           #리드 네임 목록, 논문에서는 2번만 사용해도 잘 나왔음

def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)                 # json 파일 읽어오기
    X=[]
    Y=[]
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            # if leads_names[i] == 'ii':                        #리드 번호가 2인 것들
            lead_name = leads_names[i]
            tmp = leads[lead_name]['Signal']
            tmp = np.asarray(tmp, dtype=np.float64)
            tmp = tmp / 1000.
            x.append(tmp)

        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']                       # p파, qrs파, t파 분류
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)

        y.append(background) # 0                                #그래서 y값이 원핫벡터 모양으로 나누어 나왔던 것
        y.append(p) # 1
        y.append(qrs) # 2
        y.append(t) # 3

        X.append(x)
        Y.append(y)

    X = np.array(X)
    X = np.swapaxes(X, 1, 2)

    Y = np.array(Y)
    Y = np.swapaxes(Y, 1, 2)

    return {"x":X, "y":Y}

def get_mask(table, length):
    mask = [0] * length
    for triplet in table:
        start = triplet[0]
        end = triplet[2] + 1
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i] == 0 and qrs[i] == 0 and t[i] == 0:
            background[i] = 1
    return background

