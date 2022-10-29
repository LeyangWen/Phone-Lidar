import urllib
import cv2
# from win32api import GetSystemMetrics
import numpy as np
import pickle
import glob
import os
import pandas as pd
import json

#the [x, y] for each right-click event will be stored here
left_clicks = list()
cali_index = 0
window_name = 'cali_i'


#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # todo change int to float
    global img
    global left_clicks
    global cali_index
    global window_name



    #right-click event value is 2
    if event == 1:
        # store the coordinates of the left-click event
        img = cv2.circle(img, (x, y), radius=20, color=(255, 0, 255), thickness=-1)
        # write text over the left-clicked point
        cali_index += 1
        cv2.putText(img, str(cali_index), (x+40, y), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 4)
        # img = cv2.line(img, (x, y), (4500, y), (0, 255, 0), thickness=2)
        # img = cv2.line(img, (5700, y), (8000, y), (0, 255, 0), thickness=2)
        cv2.imshow(window_name, img)
        left_clicks.append([x, y])
        print(cali_index,[x, y])


    if event == 2:
        cali_index += 1
        cv2.putText(img, str(cali_index), (x+40, y), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 255), 4)
        cv2.imshow(window_name, img)
        left_clicks.append([np.nan, np.nan])
        print(cali_index,[np.nan, np.nan])


####################################### CHANGE HERE BEFORE RUN #######################################
scene_no = 1
img_dir = f'H:\phone_Lidar\data\prelim\oct11\scene-{scene_no}'
img_extension = 'jpeg'
checkpoint_file = f'H:\phone_Lidar\data\prelim\oct11\scene-{scene_no}-2Dkps.pkl'
kp_nos = 8
kp_names = ['door_topright','door_topleft','door_bottomright','door_bottomleft','frame_topright','frame_topleft','frame_bottomright','frame_bottomleft']
####################################### CHANGE HERE BEFORE RUN #######################################
try:
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        annotation = checkpoint['annotation']
    print(f'Loaded checkpoint: {checkpoint_file}')
except:
    checkpoint = {'scene_no': scene_no, 'kp_nos':kp_nos, 'kp_names':kp_names}
    df_col = ['img_name','img_kp']
    annotation = pd.DataFrame(columns=df_col,dtype='float')
    checkpoint['annotation'] = annotation
    print(f'Created new checkpoint: {checkpoint_file}')

img_names = glob.glob(f'{img_dir}\*.{img_extension}')

img_idx = 0
while img_idx < len(img_names):
    img_name = img_names[img_idx]
    # continue loop if img_name is in checkpoint
    if img_name in checkpoint['annotation'].img_name.values.tolist():
        print(f'{img_idx}::: img_name: {img_name} is in checkpoint')
        img_idx = img_idx+1
        continue
    anno_frame = []
    img = cv2.imread(img_name)
    if isinstance(img,type(None)):
        print(f"{img_name} not found")
        img_idx = img_idx+1
        continue
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)*2 # 3 for 1 screen, 5 for 2 screen
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    window_name = f"scene_no-{scene_no}_%.4d" % img_idx
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.moveWindow(window_name, 40,30)
    print(window_name)
    #set mouse callback function for window
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, mouse_callback)
    key = cv2.waitKey(0)

    if key & 0xFF == ord('b'):
        break
    elif key & 0xFF == ord('r'):
        img_idx = img_idx - 1
    elif key & 0xFF == ord('p'):
        cali_index -=1
        left_clicks.pop()
    elif key & 0xFF == ord('c'):
        img_idx = img_idx + 1
        left_clicks = list()
        cali_index = 0
        cv2.destroyAllWindows()
        continue
    anno_frame.append(left_clicks)

    left_clicks = list()
    cali_index = 0
    cv2.destroyAllWindows()
    anno_frame = np.asarray(anno_frame[0],dtype='object')
    anno_frame = anno_frame.reshape((-1,2))
    if anno_frame.shape[0] != kp_nos:
        print(f'Try again: {img_name} has {anno_frame.shape[0]} keypoints, should be {kp_nos}')
        continue
    annotation.loc[img_idx] = [img_name,anno_frame]
    checkpoint['annotation'] = annotation
    img_idx = img_idx+1

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint,f)