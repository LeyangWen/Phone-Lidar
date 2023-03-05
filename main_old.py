import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.calibration_store import load_coefficients
import pickle
import cv2
from utils.tools import *
from matplotlib.patches import FancyArrowPatch



### Init parameters
####################################### door #######################################
scene = 1
date_dir = r'H:\phone_Lidar\data\prelim\oct31'
scene_no = 1
data_dir = os.path.join(date_dir, '2022-10-31 16_57_40')
output_dir = os.path.join(data_dir, 'output')

img_extension = 'jpeg'
checkpoint_file = f'{date_dir}\scene-{scene_no}-2Dkps.pkl'
kp_nos = 8
cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao_portrait.yml'

door_gt = np.array([[1015,1015,2128,2128,1121,1121,2181,2181]]) # mm, door l,h, frame l,h
door_sequences = [[0, 1], [3, 2], [1, 3], [2, 0], [4, 5], [7, 6], [5, 7], [6, 4]]

# flags:
display_kp = False
####################################### CHANGE HERE BEFORE RUN #######################################

####################################### mep #######################################
scene_no = 1
date_dir = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\MEP1'

data_dir = os.path.join(date_dir, 'data')
output_dir = os.path.join(date_dir, 'output')

img_extension = 'jpeg'
checkpoint_file = f'{date_dir}\MEP1-2Dkps.pkl'
kp_nos = 8
cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao_portrait.yml'

door_gt = np.array([[119,119,119,119,
                    119,119,119,119,
                    51,51,51,51]]) # mm, door l,h, frame l,h
door_sequences = [[0,1],[2,3],[4,5],[6,7],
                [0,2],[1,3],[4,6],[5,7],
                [0,4],[1,5],[2,6],[3,7]]

# flags:
display_kp = False
####################################### CHANGE HERE BEFORE RUN #######################################

####################################### Wall #######################################
scene_no = 1
img_dir = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\wall\data'

img_extension = 'jpeg'
checkpoint_file = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\wall\wall-2Dkps.pkl'
kp_nos = 5
kp_names = ['wall_top','wall_bot','wall_right','col_top','col_bot']

data_dir = os.path.join(date_dir, 'data')
output_dir = os.path.join(date_dir, 'output')

cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao_portrait.yml'

door_gt = np.array([[2885,
                     4894,
                     3025]]) # mm, door l,h, frame l,h
door_sequences = [[0,1],[1,2],[3,4]]
####################################### CHANGE HERE BEFORE RUN #######################################

####################################### M-Air #######################################
scene_no = 1
img_dir = f'H:\phone_Lidar\data\prelim\M-airFeb12\Try2.5\data'

img_extension = 'jpeg'
checkpoint_file = f'H:\phone_Lidar\data\prelim\M-airFeb12\Try2.5\M-air-2Dkps.pkl'
kp_nos = 8
kp_names = ['TopRightFront,BotRightFront,TopRightBack,BotRightBack,TopLeftFront,BotLeftFront,TopLeftBack,BotLeftBack']

cam_yml = r'H:\phone_Lidar\data\prelim\Feb14\wen14_portrait.yml'

door_gt = np.array([[3158,
                     3144,
                     3177,
                     3177,
                     5731,
                     5711,
                     11140,
                     11123
                     ]]) # mm, door l,h, frame l,h
door_sequences = [[0,1],[2,3],[4,5],[6,7],
                [1,3],[5,7],[1,5],[3,7]]
####################################### CHANGE HERE BEFORE RUN #######################################




### Read 2D keypoint annotation: pkl
with open(checkpoint_file, 'rb') as f:
    checkpoint = pickle.load(f)
    annotation = checkpoint['annotation']
    kp_names = checkpoint['kp_names']
    kp_nos = checkpoint['kp_nos']

### Read phone data: yml
camera_matrix, dist_coeffs = load_coefficients(cam_yml)

cameraPositions = []
lineP_3ds = []
depths = []
pesdoDepth = 3

figure_cam = plt.figure(200)

# arrange annotation based on index
annotation = annotation.sort_index()
door_measurements = []
for index, frame in annotation.iterrows():
    print(index)

    # read odometry json
    json_file = frame.img_name.replace(img_extension, 'json')

    # method 3: measure each frame and then average

    with open(json_file) as f:
        data = json.load(f)
        cameraEulerAngles = data['cameraEulerAngles']  # ios-ARkit XYZ Roll-Pitch-Yaw
        camera_rot_3x3M = rotation_matrix(cameraEulerAngles[0], cameraEulerAngles[1], cameraEulerAngles[2])
        cameraTransform = np.array(data['cameraTransform'][0]).T
        localToWorld = np.array(data['localToWorld']).reshape((4, 4))

        # rgb img size: 4320, 5760 ; depth img size: 192, 256
        depth_img_scale = 256/5760
        depth_map = np.array(data['depthMap'])
        depthCameraIntrinsicsInversed = np.array(data['cameraIntrinsicsInversed']).reshape((3,3))
        depth_cam_intrinsic_3x3M = np.linalg.pinv(depthCameraIntrinsicsInversed)

        cameraPosition = cameraTransform[:-1, -1]
        cameraPositions.append(cameraPosition)

    if display_kp:
        # display image and plot keypoints
        img = cv2.imread(frame.img_name)
        # plot keypoints
        for i in range(kp_nos):
            if not np.isnan(frame.img_kp[i][0]):
                img = cv2.circle(img, (int(frame.img_kp[i][0]), int(frame.img_kp[i][1])), radius=20, color=(255, 0, 255), thickness=-1)
                cv2.putText(img, kp_names[i], (int(frame.img_kp[i][0])+40, int(frame.img_kp[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 4)
        # save
        cv2.imwrite(os.path.join(output_dir, f'img_{index}.jpeg'), img)

        # display depth map and plot keypoints
        depth_img = depth_map.copy()
        # normalize depth_img to 0-255
        depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # convert gray to rgb
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)

        # plot keypoints
        for i in range(kp_nos):
            if not np.isnan(frame.img_kp[i][0]):
                depth_img = cv2.circle(depth_img, (int(frame.img_kp[i][0]*depth_img_scale), int(frame.img_kp[i][1]*depth_img_scale)), radius=2, color=(255, 0, 255), thickness=-1)
                cv2.putText(img, kp_names[i], (int(frame.img_kp[i][0]*depth_img_scale)+40, int(frame.img_kp[i][1]*depth_img_scale)), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 4)
        # save
        cv2.imwrite(os.path.join(output_dir, f'depth_{index}.jpeg'), depth_img)

    #extract depth info from depth map at keypoint location
    Lidar_depths = []
    for i in range(kp_nos):
        if not np.isnan(frame.img_kp[i][0]):
            Lidar_depth = depth_map[int(frame.img_kp[i][1]*depth_img_scale), int(frame.img_kp[i][0]*depth_img_scale)]
            Lidar_depths.append(Lidar_depth)
        else:
            Lidar_depths.append(np.nan)
    Lidar_depths = np.array(Lidar_depths)


    figure_cam,ax_cam = draw_camera(localToWorld.T, camera_matrix, figure = figure_cam, cameraName=index)

    kp_no = frame.img_kp.shape[0]

    #  landscape keypoints to portrait keypoints
    # frame.img_kp = np.array([[0,0],[0,4320],[5760,0],[5760,4320]]) # test cases
    # rot_kp = np.array([[0, -1], [1, 0]]).dot(frame.img_kp.T).T + np.array([4320, 0])
    rot_kp = frame.img_kp
    dist_kp = rot_kp.astype(np.float32).reshape((kp_no, 1, 2))
    undist_kp = cv2.undistortPoints(dist_kp, camera_matrix, dist_coeffs, P=camera_matrix).reshape((kp_no, 2))

    localPoint = (np.linalg.pinv(camera_matrix) @
                  np.hstack(
                      (undist_kp, np.ones((kp_no, 1)))
                  ).T
                  ).T * (Lidar_depths.reshape((kp_no, 1)))
    lineP_3d = (localToWorld.T @
                np.hstack(
                    (
                        localPoint, np.ones((kp_no, 1))
                    )
                ).T
                ).T

    lineP_3d = np.array([np.divide(lineP_3d[:, 0], lineP_3d[:, 3].T), np.divide(lineP_3d[:, 1], lineP_3d[:, 3].T),
                         np.divide(lineP_3d[:, 2], lineP_3d[:, 3].T)]).T
    lineP_3ds.append(lineP_3d)
    depth = np.linalg.norm(lineP_3d - cameraPosition, axis=1)
    depths.append(depth)
    this_measurement = measure_obj(lineP_3d, door_sequences)
    door_measurements.append(this_measurement)


depths = np.array(depths)
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
    ax.scatter(cameraPositions[i, 0], cameraPositions[i, 1], cameraPositions[i, 2], s=10, c='r')
    ax.text(cameraPositions[i, 0] + text_offset, cameraPositions[i, 1] + text_offset,
            cameraPositions[i, 2] + text_offset, f"frame_{i}", color='black', fontsize=15)
    for kp in range(kp_nos):
        ax.plot([cameraPositions[i, 0], lineP_3ds[i, kp, 0]], [cameraPositions[i, 1], lineP_3ds[i, kp, 1]],
                [cameraPositions[i, 2], lineP_3ds[i, kp, 2]], c='c')
        ax.text(lineP_3ds[i, kp, 0] + text_offset / 2 * kp, lineP_3ds[i, kp, 1] + text_offset / 2 * kp,
                lineP_3ds[i, kp, 2] + text_offset / 2 * kp, kp_names[kp], size=10, zorder=1, color='k', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# # viewing angle -90. -60
# # ax.view_init(elev=-60., azim=-90)
# # equal aspect ratio
# # ax.set_aspect('equal')
# plt.savefig(f'figures\scene-{scene_no}-3Dkps-{fig}.png')

est_kps = np.zeros((kp_nos, 3))
color_8 = ['r', 'g', 'b', 'm','r', 'g', 'b', 'm']
for kp in range(kp_nos):
    fig = plt.figure(kp)
    ax = fig.add_subplot(111, projection='3d')

    P0 = []
    P1 = []

    # plot line from camera to first 3D point

    for i in range(len(cameraPositions)):
        if depths[i, kp] > 6:
            continue
        alpha = i / len(cameraPositions)
        if lineP_3ds[i, kp, 0] == np.nan:
            continue
        ax.scatter(cameraPositions[i, 0], cameraPositions[i, 1], cameraPositions[i, 2], s=4, c='r', alpha=alpha)
        ax.plot([cameraPositions[i, 0], lineP_3ds[i, kp, 0]], [cameraPositions[i, 1], lineP_3ds[i, kp, 1]],
                [cameraPositions[i, 2], lineP_3ds[i, kp, 2]], c='c', alpha=alpha / 3)
        ax.scatter(lineP_3ds[i, kp, 0], lineP_3ds[i, kp, 1], lineP_3ds[i, kp, 2], s=4, c='g', alpha=alpha)

        if not (np.isnan(lineP_3ds[i, kp, 0]) or np.isnan(lineP_3ds[i, kp, 1]) or np.isnan(lineP_3ds[i, kp, 2])):
            P0.append(cameraPositions[i])
            P1.append(lineP_3ds[i, kp])

    P0 = np.array(P0)
    P1 = np.array(P1)
    ### find best-fit 3D point
    # method 1: rough intersection of 3D lines
    est_kp = intersect(P0, P1)
    # method 2: use Lidar depth and RANSAC
    # est_kp = pts_center_ransac(P1)

    est_kps[kp] = est_kp.reshape(3)
    ax.scatter(est_kp[0], est_kp[1], est_kp[2], s=28, color='blue', marker='P')
    ax_cam.scatter(est_kp[0], est_kp[1], est_kp[2], s=28, color=color_8[kp], marker='P')
    # ax_cam.text(est_kp[0], est_kp[1] , est_kp[2], kp, color='black')

    # show axis label

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


### Get 3D lines


### Calculate 3D dim & pos
# Finding the intersection point of many lines in 3D

door_dists = np.zeros((len(door_sequences),1))
for seq_id, seq in enumerate(door_sequences):
    door_dists[seq_id] = dist(est_kps[seq[0]], est_kps[seq[1]])
    ax_cam.plot([est_kps[seq[0], 0], est_kps[seq[1], 0]], [est_kps[seq[0], 1], est_kps[seq[1], 1]],
                [est_kps[seq[0], 2], est_kps[seq[1], 2]], c='c', alpha=1)

print(f'door_dists: {door_dists*1000}')

# method 3: frame wise mean/median
door_measurements = np.array(door_measurements)
door_dists = np.nanmedian(door_measurements, axis=0)
print(f'door_dists_median: \n{door_dists*1000}')

door_dists = np.nanmean(door_measurements, axis=0)
print(f'door_dists_mean: \n{door_dists*1000}')


plt.show()