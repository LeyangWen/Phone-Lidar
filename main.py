import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.calibration_store import load_coefficients
import pickle

### Init parameters
date_dir = r'H:\phone_Lidar\data\prelim\oct11'
scene_no = 1
data_dir = os.path.join(date_dir, f'scene-{scene_no}')
cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao.yml'

### Read phone data: yml
camera_matrix, dist_coeffs = load_coefficients(cam_yml)

for frame in xxx:
    pass
# read json
json_file = os.path.join(data_dir, 'phone_data.json')
with open(json_file) as f:
    data = json.load(f)

### Read 2D keypoints: pkl

### Get 3D lines

### find best-fit 3D point

### Calculate 3D dim & pos