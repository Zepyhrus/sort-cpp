"""
Calculate precision and recall
"""
import os
from os.path import join, split


import numpy as np
import numpy.random as npr
import pandas as pd


sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof',
             'ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17',
             'ADL-Rundle-6','ADL-Rundle-8','Venice-2']

OUTPUT_DIR = 'output'
LABEL_DIR = 'mot_benchmark/train'

target_seq = sequences[0]

result = pd.read_csv(join(OUTPUT_DIR, target_seq+'.txt'), sep=',', header=None)
label = pd.read_csv(
  join(LABEL_DIR, target_seq, 'gt/gt.txt'), sep=',', header=None)

print(len(np.unique(result[1])))
print('-'*20)
print(len(np.unique(label[1])))
