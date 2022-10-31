import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.calibration_store import load_coefficients
import pickle
from utils.tools import *
from matplotlib.patches import FancyArrowPatch

# todo: check if rotation matrix is correct
# todo: delete blured calibration images
# todo: verifiy if odometry is correct using calibration images

### Init parameters
####################################### CHANGE HERE BEFORE RUN #######################################
scene = 1
date_dir = r'H:\phone_Lidar\data\prelim\oct11'
scene_no = 1
data_dir = os.path.join(date_dir, f'scene-{scene_no}')
cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao_portrait.yml'
img_extension = 'jpeg'
checkpoint_file = f'{date_dir}\scene-{scene_no}-2Dkps.pkl'
kp_nos = 8
####################################### CHANGE HERE BEFORE RUN #######################################

### Read 2D keypoint annotation: pkl
with open(checkpoint_file, 'rb') as f:
    checkpoint = pickle.load(f)
    annotation = checkpoint['annotation']
    kp_names = checkpoint['kp_names']
    kp_nos = checkpoint['kp_nos']

### Read phone data: yml
camera_matrix, dist_coeffs = load_coefficients(cam_yml)
camera_matrix_4x4 = np.eye(4)
camera_matrix_4x4[:3, :3] = camera_matrix
# fixme: combine dist_coeffs and camera_matrix_4x4 into a single matrix


cameraPositions = []
lineP_3ds = []
pesdoDepth = -2.5
for index, frame in annotation.iterrows():
    # frame.name
    # frame.img_name
    # frame.img_kp

    # read odometry json
    json_file = frame.img_name.replace(img_extension, 'json')
    with open(json_file) as f:
        data = json.load(f)
        cameraEulerAngles = data['cameraEulerAngles'] # ios-ARkit XYZ Roll-Pitch-Yaw
        camera_rot_M = rotation_matrix(cameraEulerAngles[0], cameraEulerAngles[1], cameraEulerAngles[2])
        cameraTransform = np.array(data['cameraTransform'][0])
        cameraTransform[:-1,:-1] = camera_rot_M
        cameraPosition = cameraTransform[-1,:-1]
        cameraPositions.append(cameraPosition)




    kp_no = frame.img_kp.shape[0]
    #     add 1 & 1/z to the last column of img_kp
    #     https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
    # https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/gravity


    #  this is portrait
    rot_kp = np.array([[0,1],[1,0]]).dot(frame.img_kp.T).T+np.array([4320,0])
    img_kp = np.hstack((rot_kp, np.ones((kp_no, 1)), 1/pesdoDepth*np.ones((kp_no, 1))))
    # todo: redo calibration w. rotated img

    # this is landscape (annotation is landscape)
    # img_kp = np.hstack((frame.img_kp, np.ones((kp_no, 1)), 1/pesdoDepth*np.ones((kp_no, 1))))

    lineP_3d = np.dot(np.linalg.pinv(1/pesdoDepth*np.dot(camera_matrix_4x4, cameraTransform.T)),img_kp.T).T
    lineP_3d = np.array([np.divide(lineP_3d[:,0],lineP_3d[:,3].T),np.divide(lineP_3d[:,1],lineP_3d[:,3].T),np.divide(lineP_3d[:,2],lineP_3d[:,3].T)]).T
    lineP_3ds.append(lineP_3d)



cameraPositions = np.array(cameraPositions)
lineP_3ds = np.array(lineP_3ds)
### Plot
fig = plt.figure(100)
ax = fig.add_subplot(111, projection='3d')
# 8 different colors
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
text_offset = 0.05
for i in range(1):
    # if i%100 != 0:
    #     continue
    ax.scatter(cameraPositions[i,0], cameraPositions[i,1], cameraPositions[i,2], s=10, c='r')
    ax.text(cameraPositions[i,0]+text_offset, cameraPositions[i,1]+text_offset, cameraPositions[i,2]+text_offset, f"frame_{i}", color='black', fontsize=15)
    for kp in range(8):
        ax.plot([cameraPositions[i,0], lineP_3ds[i,kp,0]], [cameraPositions[i,1], lineP_3ds[i,kp,1]], [cameraPositions[i,2], lineP_3ds[i,kp,2]], c='c')
        ax.text(lineP_3ds[i,kp,0]+text_offset/2*kp, lineP_3ds[i,kp,1]+text_offset/2*kp, lineP_3ds[i,kp,2]+text_offset/2*kp, kp_names[kp], size=10, zorder=1, color='k', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# # viewing angle -90. -60
# # ax.view_init(elev=-60., azim=-90)
# # equal aspect ratio
# # ax.set_aspect('equal')
# plt.savefig(f'figures\scene-{scene_no}-3Dkps-{fig}.png')


for kp in range(kp_nos):
    fig = plt.figure(kp)
    ax = fig.add_subplot(111, projection='3d')

    # plot line from camera to first 3D point
    for i in range(len(cameraPositions)):
        if lineP_3ds[i,kp,0] == np.nan:
            continue
        ax.scatter(cameraPositions[i,0], cameraPositions[i,1], cameraPositions[i,2], s=4, c='r')
        ax.plot([cameraPositions[i,0], lineP_3ds[i,kp,0]], [cameraPositions[i,1], lineP_3ds[i,kp,1]], [cameraPositions[i,2], lineP_3ds[i,kp,2]], c='c')
    # show axis label

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
plt.show()

### Get 3D lines


### find best-fit 3D point

### Calculate 3D dim & pos
# Finding the intersection point of many lines in 3D



# rough intersection of 3D lines
