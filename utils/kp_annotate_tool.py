import urllib
import cv2
# from win32api import GetSystemMetrics
import numpy as np
import pickle
import glob
import os
import pandas as pd
import json
import yaml

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
        cross_size = 250
        img = cv2.line(img, (x-cross_size, y), (x+cross_size, y), (0, 255, 0), thickness=2)
        img = cv2.line(img, (x, y-cross_size), (x, y+cross_size), (0, 255, 0), thickness=2)
        left_clicks.append([x, y])
        print(cali_index,[x, y])
        cv2.imshow(window_name, img)


    if event == 2:
        cali_index += 1
        cv2.putText(img, str(cali_index), (x+40, y), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 255), 4)
        cv2.imshow(window_name, img)
        left_clicks.append([np.nan, np.nan])
        print(cali_index,[np.nan, np.nan])


# ####################################### door #######################################
# scene_no = 1
# img_dir = f'H:\phone_Lidar\data\prelim\oct31\\2022-10-31 16_57_40\data'
#
# img_extension = 'jpeg'
# checkpoint_file = f'H:\phone_Lidar\data\prelim\oct31\scene-{scene_no}-2Dkps.pkl'
# kp_nos = 8
# kp_names = ['door_topright','door_topleft','door_bottomright','door_bottomleft','frame_topright','frame_topleft','frame_bottomright','frame_bottomleft']
# ####################################### CHANGE HERE BEFORE RUN #######################################
#
# ####################################### MEP1 #######################################
# scene_no = 1
# img_dir = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\MEP1\data'
#
# img_extension = 'jpeg'
# checkpoint_file = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\MEP1\MEP1-2Dkps.pkl'
# kp_nos = 8
# kp_names = ['MEP_topright','MEP_topleft','MEP_bottomright','MEP_bottomleft','wall_topright','wall_topleft','wall_bottomright','wall_bottomleft']
# ####################################### CHANGE HERE BEFORE RUN #######################################
#
# ####################################### Wall #######################################
# scene_no = 1
# img_dir = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\wall\data'
#
# img_extension = 'jpeg'
# checkpoint_file = f'H:\phone_Lidar\data\prelim\\Nov21\\Nov21\wall\wall-2Dkps.pkl'
# kp_nos = 5
# kp_names = ['wall_top','wall_bot','wall_right','col_top','col_bot']
# ####################################### CHANGE HERE BEFORE RUN #######################################
#
# ####################################### M-Air #######################################
# scene_no = 1
# img_dir = f'H:\phone_Lidar\data\prelim\M-airFeb12\Try2.5\data'
#
# img_extension = 'jpeg'
# checkpoint_file = f'H:\phone_Lidar\data\prelim\M-airFeb12\Try2.5\M-air-2Dkps.pkl'
# kp_nos = 8
# kp_names = ['TopRightFront','BotRightFront','TopRightBack','BotRightBack','TopLeftFront','BotLeftFront','TopLeftBack','BotLeftBack']
# ####################################### CHANGE HERE BEFORE RUN #######################################
base_dir = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\Phone-Lidar'
config_file = f'{base_dir}/config/1_init_test/door.yaml'
config_file = f'{base_dir}/config/1_init_test/Mair.yaml'
config_file = f'{base_dir}/config/1_init_test/MEPbox.yaml'
config_file = f'{base_dir}/config/2_odometry_check/door.yaml'
config_file = f'{base_dir}/config/2_odometry_check/door_w_stabilizer.yaml'


with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    checkpoint_file = config['checkpoint_file']
    img_dir = config['data_dir']
    img_extension = config['img_extension']
    kp_nos = config['kp_nos']
    kp_names = config['kp_names']
    checkpoint_file = config['checkpoint_file']
    scene_no = 1


try:
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        annotation = checkpoint['annotation']
    print(f'Loaded checkpoint: {checkpoint_file}, {len(annotation)} images already annotated')
except:
    checkpoint = {'scene_no': scene_no, 'kp_nos': kp_nos, 'kp_names': kp_names}
    df_col = ['img_name', 'img_kp']
    annotation = pd.DataFrame(columns=df_col, dtype='object')
    annotation['img_name'] = annotation['img_name'].astype(str)
    annotation['img_kp'] = annotation['img_kp'].astype(object)
    checkpoint['annotation'] = annotation
    print(f'Created new checkpoint: {checkpoint_file}')

img_names = sorted(glob.glob(f'{img_dir}\*.{img_extension}'))

img_idx = 0
count = 0
while img_idx < len(img_names):
    img_name = img_names[img_idx]
    # continue loop if img_name is in checkpoint
    print(f'{img_idx}/{len(img_names)}', img_name)
    if img_name in checkpoint['annotation'].img_name.values.tolist() or img_name.replace('Y:','H:') in checkpoint['annotation'].img_name.values.tolist():
        print(f'{img_idx}::: img_name: {img_name} is in checkpoint')
        img_idx = img_idx+1
        count += 1
        continue
    # if img_idx % 2 != 1: # use this to control how many images to annotate
    #     img_idx = img_idx+1
    #     continue
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

    if key & 0xFF == ord('b'):  # press b to stop
        cv2.destroyAllWindows()
        break
    elif key & 0xFF == ord('r'):  # press r to go back
        img_idx = img_idx - 1
    elif key & 0xFF == ord('p'):  # press p to redo last point
        cali_index -=1
        left_clicks.pop()
    elif key & 0xFF == ord('c'):  # press c to skip
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
    annotation.loc[img_idx] = [img_name, anno_frame]
    checkpoint['annotation'] = annotation
    img_idx = img_idx+1

    count += 1
    # annotation.img_name = annotation.img_name.str.slice_replace(stop = 58, repl='H:\phone_Lidar\data\prelim\oct31\\2022-10-31 16_57_40\data\\1')
    # annotation.img_name[1]
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint,f)

print(f'Finished: {count} images annotated')