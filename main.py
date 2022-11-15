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

# todo: check if rotation matrix is correct
# todo: delete blured calibration images
# todo: verifiy if odometry is correct using calibration images

### Init parameters
####################################### CHANGE HERE BEFORE RUN #######################################
scene = 1
date_dir = r'H:\phone_Lidar\data\prelim\oct31'
scene_no = 1
data_dir = os.path.join(date_dir, f'scene-{scene_no}')

img_extension = 'jpeg'
checkpoint_file = f'{date_dir}\scene-{scene_no}-2Dkps.pkl'
kp_nos = 8
portrait_flag = True
oct31Test = True
if portrait_flag:
    cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao_portrait.yml'
else:
    cam_yml = r'H:\phone_Lidar\data\prelim\oct11\Hongxiao.yml'
####################################### CHANGE HERE BEFORE RUN #######################################

### Read 2D keypoint annotation: pkl
with open(checkpoint_file, 'rb') as f:
    checkpoint = pickle.load(f)
    annotation = checkpoint['annotation']
    kp_names = checkpoint['kp_names']
    kp_nos = checkpoint['kp_nos']

### Read phone data: yml
camera_matrix, dist_coeffs = load_coefficients(cam_yml)
cam_intrinsic_4x4M = np.eye(4)
cam_intrinsic_4x4M[:3, :3] = camera_matrix


cameraPositions = []
lineP_3ds = []
depths = []
pesdoDepth = 3

figure = plt.figure()


# arrange annotation based on index
annotation = annotation.sort_index()

for index, frame in annotation.iterrows():
    print(index)
    # if index > 111:
    #     break
    # frame.name
    # frame.img_name
    # frame.img_kp

    # # display image and plot keypoints
    # img = cv2.imread(frame.img_name)
    # # plot keypoints
    # for i in range(kp_nos):
    #     if not np.isnan(frame.img_kp[i][0]):
    #         img = cv2.circle(img, (int(frame.img_kp[i][0]), int(frame.img_kp[i][1])), radius=20, color=(255, 0, 255), thickness=-1)
    #         cv2.putText(img, kp_names[i], (int(frame.img_kp[i][0])+40, int(frame.img_kp[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 4)
    # # save
    # cv2.imwrite(os.path.join(data_dir,'annotation', f'img_{index}.jpeg'), img)

    # read odometry json
    json_file = frame.img_name.replace(img_extension, 'json')
    with open(json_file) as f:
        data = json.load(f)
        cameraEulerAngles = data['cameraEulerAngles'] # ios-ARkit XYZ Roll-Pitch-Yaw
        camera_rot_3x3M = rotation_matrix(cameraEulerAngles[0], cameraEulerAngles[1], cameraEulerAngles[2])
        cameraTransform = np.array(data['cameraTransform'][0]).T
        if oct31Test:
            cameraIntrinsicsInversed = np.array(data['cameraIntrinsicsInversed']).reshape((3,3))
            localToWorld = np.array(data['localToWorld']).reshape((4,4))
            depth_cam_intrinsic_3x3M = np.linalg.pinv(cameraIntrinsicsInversed) # not sure

        if True:
            # if ios case is set to gravity (0)
            camera_rot_4x4M = np.eye(4)
            camera_rot_4x4M[:3, :3] = rotation_matrix(-np.pi, 0, 0) @ camera_rot_3x3M
            camera_transform_4x4M = np.eye(4)
            camera_transform_4x4M[:3, 3] = cameraTransform[:3, 3]
            cam_extrinsic_4x4M = camera_transform_4x4M @ camera_rot_4x4M


            cameraTransform[:-1,:-1] = (camera_rot_3x3M.T) @ (cameraTransform[:-1,:-1]) #

        cameraPosition = cameraTransform[:-1,-1]
        cameraPositions.append(cameraPosition)


        draw_camera(localToWorld.T, camera_matrix, figure = figure, cameraName=index)
        # break
        # if index == 91:
        #     break
























    # set img kp to center of image /2 5760 4320
    # frame.img_kp = np.array([[5760/2, 4320/2],[5760/2, 4320/2],[5760/2, 4320/2],[5760/2, 4320/2],
    #                          [5760/2, 4320/2],[5760/2, 4320/2],[5760/2, 4320/2],[5760/2, 4320/2],])

    kp_no = frame.img_kp.shape[0]
    #     add 1 & 1/z to the last column of img_kp
    #     https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
    # https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/gravity

    if portrait_flag:
        #  this is portrait
        rot_kp = np.array([[0,1],[1,0]]).dot(frame.img_kp.T).T+np.array([4320,0])
        dist_kp = rot_kp.astype(np.float32).reshape((kp_no,1,  2))
        undist_kp = cv2.undistortPoints(dist_kp, camera_matrix, dist_coeffs, P=camera_matrix).reshape((kp_no, 2))


        img_kp = np.hstack((rot_kp, np.ones((kp_no, 1)), 1/pesdoDepth*np.ones((kp_no, 1))))
    else:
        # this is landscape (annotation is landscape)
        dist_kp = frame.img_kp.astype(np.float32).reshape((kp_no,1,  2))
        undist_kp = cv2.undistortPoints(dist_kp, camera_matrix, dist_coeffs, P=camera_matrix).reshape((kp_no, 2))
        img_kp = np.hstack((undist_kp, np.ones((kp_no, 1)), 1/pesdoDepth*np.ones((kp_no, 1))))
    #     why 1/pesdoDepth: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
    if not oct31Test:
        # lineP_3d = np.dot(np.linalg.pinv(1 / pesdoDepth * np.dot(cam_intrinsic_4x4M, cam_extrinsic_4x4M)), img_kp.T).T
        lineP_3d = (
            np.linalg.pinv(1 / pesdoDepth *
                           (cam_intrinsic_4x4M @ cam_extrinsic_4x4M)) @ img_kp.T
        ).T

    else:
        localPoint = (np.linalg.pinv(camera_matrix) @
                      np.hstack(
                          (undist_kp, np.ones((kp_no, 1)))
                                ).T
                      ).T*pesdoDepth
        lineP_3d = (localToWorld.T @
            np.hstack(
                (
                    localPoint, np.ones((kp_no, 1))
                )
            ).T
        ).T


    lineP_3d = np.array([np.divide(lineP_3d[:,0],lineP_3d[:,3].T),np.divide(lineP_3d[:,1],lineP_3d[:,3].T),np.divide(lineP_3d[:,2],lineP_3d[:,3].T)]).T
    lineP_3ds.append(lineP_3d)
    depth = np.linalg.norm(lineP_3d-cameraPosition, axis=1)
    depths.append(depth)


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
    # fixme: this still look weird
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
        if depths[i,kp] > 6:
            continue
        alpha = i/len(cameraPositions)
        if lineP_3ds[i,kp,0] == np.nan:
            continue
        ax.scatter(cameraPositions[i,0], cameraPositions[i,1], cameraPositions[i,2], s=4, c='r', alpha=alpha)
        ax.plot([cameraPositions[i,0], lineP_3ds[i,kp,0]], [cameraPositions[i,1], lineP_3ds[i,kp,1]], [cameraPositions[i,2], lineP_3ds[i,kp,2]], c='c', alpha=alpha/3)
        ax.scatter(lineP_3ds[i,kp,0], lineP_3ds[i,kp,1], lineP_3ds[i,kp,2], s=4, c='g', alpha=alpha)
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
