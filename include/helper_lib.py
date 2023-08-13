import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from PIL import Image
import ultralytics
from ultralytics import YOLO
import yaml

def prepare_images(img_list, ano_paths, ts_path, dest_path):
    for i in img_list:
        ano_path = ano_paths[i]
        img_path = os.path.join(ts_path, ano_path.split('/')[-1][0:-4]+'.jpg')
        try:
            shutil.copy(ano_path, dest_path)
            shutil.copy(img_path, dest_path)
        except:
            continue

def choose_train_valid_test(ts_path):
    ano_paths = []
    for dirname, _, filenames in os.walk(ts_path):
        for filename in filenames:
            ano_paths += [(os.path.join(dirname, filename))]
            
    n = 600
    N = list(range(n))
    random.shuffle(N)

    train_ratio = 0.7
    valid_ratio = 0.2

    train_size = int(train_ratio*n)
    valid_size = int(valid_ratio*n)

    train_i = N[:train_size]
    valid_i = N[train_size:train_size+valid_size]
    test_i = N[train_size+valid_size:]

    return train_i, valid_i, test_i, ano_paths

def create_data_yaml():
    data_yaml = dict(
        train ='train',
        val ='valid',
        test ='test',
        nc = 4,
        names = ['prohibitor','danger','mandatory','other']
    )
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)