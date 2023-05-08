import numpy as np
import pickle
from utils.calibration_store import load_coefficients
import cv2
from utils.tools import *
import yaml
import json


# class
class PhoneLidar():
    def __init__(self, config_file):
        self.config_file = config_file
        self.filter_range = 4
        self.__load_config()
        self.__load_2Dkps()
        self.__load_calib()
        self.run()

    def run(self):
        self.iter_frames()
        self.iter_kps()
        self.print_measurements()

    def print_measurements(self):
        for i in range(4):
            print()
            if i == 0:
                measurements = self.intersect_measurements
                print(f'method_1_intersect:')
            elif i == 1:
                measurements = self.ransac_measurements
                print(f'method_2_RANSAC:, filter_range={self.filter_range}')
            elif i == 2:
                measurements = np.nanmedian(self.inFrame_measurements, axis=0)#, weights=self.measurement_weights)
                print(f'method_3_median:')
            elif i == 3:
                measurements = np.nanmean(self.inFrame_measurements, axis=0)#, weights=self.measurement_weights)
                print(f'method_4_mean:')
            gt_measurements = self.config['dist_gt']
            difference = compare_gt(measurements.reshape((-1))*1000, gt_measurements)
            print(f'gt,measure,diff,percentage%')
            diffs = []
            percentages = []
            for measure, gt, diff in zip(measurements.reshape((-1)), gt_measurements, difference):
                # format into two decimal places
                measure = np.round(measure*1000, 2)
                gt = np.round(gt, 2)
                diff = np.round(diff, 2)
                percentage = np.round(diff/gt*100, 2)
                print(f'{gt},{measure},{diff},{percentage}%')
                diffs.append(diff)
                percentages.append(percentage)
            print(f'Average, ,{np.round(np.mean(diffs), 2)}, {np.round(np.mean(percentages), 2)}%')

    def __load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.checkpoint_file = self.config['checkpoint_file']

    def __load_calib(self):
        self.camera_matrix, self.dist_coeffs = load_coefficients(self.config['cam_yml'])

    def __load_2Dkps(self):
        with open(self.checkpoint_file, 'rb') as f:
            self.checkpoint = pickle.load(f)
            self.annotation = self.checkpoint['annotation'].sort_index()
            self.kp_names = self.checkpoint['kp_names']
            self.kp_nos = self.checkpoint['kp_nos']

    def load_lidar(self, frame_no):
        # read odometry json
        frame = self.annotation.iloc[frame_no]
        json_file = frame.img_name.replace(self.config['img_extension'], 'json').replace('H:', 'Y:')
        with open(json_file) as f:
            data = json.load(f)
            cameraEulerAngles = data['cameraEulerAngles']  # ios-ARkit XYZ Roll-Pitch-Yaw
            camera_rot_3x3M = rotation_matrix(cameraEulerAngles[0], cameraEulerAngles[1], cameraEulerAngles[2])
            cameraTransform = np.array(data['cameraTransform'][0]).T
            localToWorld = np.array(data['localToWorld']).reshape((4, 4))

            # rgb img size: 4320, 5760 ; depth img size: 192, 256
            depth_map = np.array(data['depthMap'])
            depthCameraIntrinsicsInversed = np.array(data['cameraIntrinsicsInversed']).reshape((3, 3))
            depth_cam_intrinsic_3x3M = np.linalg.pinv(depthCameraIntrinsicsInversed)
            cameraPosition = cameraTransform[:-1, -1]
        return cameraPosition, depth_map, localToWorld, camera_rot_3x3M, depth_cam_intrinsic_3x3M

    def iter_frames(self):
        self.cameraPositions = []
        self.depthMaps = []
        self.localToWorlds = []
        self.Lidar_depths = []
        self.lineP_3ds = []
        self.inFrame_measurements = []
        self.weights = []
        self.measurement_weights = []
        for frame_no, [frame_idx, frame] in enumerate(self.annotation.iterrows()):
            print(f'frame {frame_no}/{len(self.annotation)}', end='\r')
            cameraPosition, depth_map, localToWorld, camera_rot_3x3M, depth_cam_intrinsic_3x3M = self.load_lidar(
                frame_no)
            self.cameraPositions.append(cameraPosition)
            self.depthMaps.append(depth_map)
            self.localToWorlds.append(localToWorld)
            Lidar_depth, weight = self.extract_kps_depth(frame_no, filter_range=self.filter_range)
            self.Lidar_depths.append(Lidar_depth)
            self.weights.append(weight)
            lineP_3d = self.project_kps(frame_no)
            self.lineP_3ds.append(lineP_3d)
            # method 3
            self.inFrame_measurements.append(measure_obj(lineP_3d, self.config['dist_sequences']))
            self.measurement_weights.append(measure_obj(lineP_3d, self.config['dist_sequences']))
        self.inFrame_measurements = np.array(self.inFrame_measurements)
        self.lineP_3ds = np.array(self.lineP_3ds)
        self.weights = np.array(self.weights)

    def iter_kps(self):
        self.est_kps1 = np.zeros((self.kp_nos, 3))
        self.est_kps2 = np.zeros((self.kp_nos, 3))
        for kp in range(self.kp_nos):
            print(f'keypoint {kp}/{self.kp_nos}', end='\r')
            P0 = []
            P1 = []
            weights = []
            lineP_3ds = self.lineP_3ds
            for i in range(len(self.cameraPositions)):
                if not (np.isnan(lineP_3ds[i][kp, 0])
                        or np.isnan(lineP_3ds[i][kp, 1])
                        or np.isnan(lineP_3ds[i][kp, 2])):
                    P0.append(self.cameraPositions[i])
                    P1.append(self.lineP_3ds[i, kp])
                    weights.append(self.weights[i, kp])
            ### find best-fit 3D point
            P0 = np.array(P0)
            P1 = np.array(P1)
            # method 1: rough intersection of 3D lines
            est_kp1 = intersect(P0, P1)
            # method 2: use Lidar depth and RANSAC
            est_kp2 = pts_center_ransac(P1)#, weights=weights)
            self.est_kps1[kp] = est_kp1.reshape(3)
            self.est_kps2[kp] = est_kp2.reshape(3)
        self.intersect_measurements = measure_obj(self.est_kps1,self.config['dist_sequences'])
        self.ransac_measurements = measure_obj(self.est_kps2,self.config['dist_sequences'])

    def extract_kps_depth(self, frame_no, filter_range = 3):
        # filter_range: 5m, set to 0 for no filter
        # filter_size: (2n+1)x(2n+1) smooth convolution filter, set to 0 for no filter
        # rgb img size: 4320, 5760 ; depth img size: 192, 256
        depth_img_scale = 256 / 5760
        frame = self.annotation.iloc[frame_no]
        depth_map = self.depthMaps[frame_no]
        Lidar_depth = np.zeros(self.kp_nos)
        weight = np.ones(self.kp_nos)
        for i in range(self.kp_nos):
            if np.isnan(frame.img_kp[i][1]) or np.isnan(frame.img_kp[i][0]):
                Lidar_depth[i] = np.nan
                continue
            x, y = int(np.rint(frame.img_kp[i][1] * depth_img_scale)), int(np.rint(frame.img_kp[i][0] * depth_img_scale))
            # get filter_size x filter_size depth map
            filter_size = 0  # made it worse, deprecated
            block = depth_map[x - filter_size:x + filter_size + 1, y - filter_size:y + filter_size + 1]
            if filter_range != 0:
                block = block[block < filter_range]
                kp_weight = np.exp(-block / filter_range)/np.exp(1)  # exp(-x/range)
                # kp_weight = np.exp(-(block / filter_range)**2)  # exp(-(x/range)^2)
                # kp_weight = (block - filter_range)**2 / 4  # 0.25(x-5)^2
                weight[i] = np.nanmean(kp_weight)
            Lidar_depth[i] = np.nanmean(block)
        return Lidar_depth, weight

    def project_kps(self, frame_no):
        # project kps to 3D
        frame = self.annotation.iloc[frame_no]
        # cameraPosition = self.cameraPositions[frame_no]
        localToWorld = self.localToWorlds[frame_no]
        # camera_rot_3x3M = self.camera_rot_3x3M[frame_no]
        # depth_cam_intrinsic_3x3M = self.depth_cam_intrinsic_3x3M[frame_no]
        Lidar_depths = self.Lidar_depths[frame_no]
        kp_no = self.kp_nos
        rot_kp = frame.img_kp
        dist_kp = rot_kp.astype(np.float32).reshape((kp_no, 1, 2))
        undist_kp = cv2.undistortPoints(dist_kp, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape((kp_no, 2))

        localPoint = (np.linalg.pinv(self.camera_matrix) @
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

        # depth = np.linalg.norm(lineP_3d - cameraPosition, axis=1) # just for checking
        return lineP_3d


class PhoneLidarCheckerboardValidate(PhoneLidar):
    def __init__(self, config_file):
        self.square_size = 0.0288  # 2.88cm
        self.width = 5
        self.height = 7
        super().__init__(config_file)
        self.config_file = config_file
        frame_no = 32
        (self.get_gt_camera_pose(self.get_checkerboard(frame_no)))
        print()
        (self.load_lidar(frame_no))

    # def run(self):
    #     print('overriding run()')

    def get_checkerboard(self, frame_no, show_img=False):
        frame = self.annotation.iloc[frame_no]
        img_name = frame.img_name
        # img_name = r'Y:\phone_Lidar\data\2_odometry_check\2023-04-19_060107\data\110454.986576833_5.jpeg'
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
        if ret:
            corners = corners.reshape(-1, 2)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if show_img:
                img = cv2.drawChessboardCorners(img, (self.width, self.height), corners2, ret)
                # write numbers on corners
                for i in range(corners2.shape[0]):
                    img = cv2.putText(img, str(i), tuple(corners2[i].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                scale_width = 640 / img.shape[1]
                scale_height = 480 / img.shape[0]
                scale = min(scale_width, scale_height) * 2  # 3 for 1 screen, 5 for 2 screen
                window_width = int(img.shape[1] * scale)
                window_height = int(img.shape[0] * scale)
                window_name = f"img_%.4d" % frame_no
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, window_width, window_height)
                cv2.moveWindow(window_name, 40, 30)
                cv2.imshow(window_name, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return corners2
        else:
            return ret

    def get_gt_camera_pose(self, carpet_2D):
        # solve pnp
        carpet_3D = np.zeros((self.width * self.height, 3), np.float32)
        for w in range(self.width):
            for h in range(self.height):
                carpet_3D[h * self.width + w, :] = np.array([self.width-w-1, h, 0])
        # print(carpet_3D)
        carpet_3D *= self.square_size
        success, rotation_vector, translation_vector = cv2.solvePnP(carpet_3D, carpet_2D, self.camera_matrix, self.dist_coeffs, flags=0)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        RT4x4 = np.eye(4)
        RT4x4[:3, :3] = rotation_matrix
        RT4x4[:3, 3] = translation_vector.reshape((3,))
        # RT4x4[:3, 3] = -rotation_matrix.dot(translation_vector.reshape((3,)))
        return success, rotation_matrix, translation_vector, rotation_vector, RT4x4.T

    def iter_frames(self):
        self.cameraPositions = []
        self.depthMaps = []
        self.localToWorlds = []
        self.Lidar_depths = []
        self.lineP_3ds = []
        self.inFrame_measurements = []
        self.weights = []
        self.measurement_weights = []
        figure_gt_cam_pos = plt.figure('GT checkerboard camera position')
        ax_cam = figure_gt_cam_pos.add_subplot(111, projection='3d')
        figure_lidar_cam_pos = plt.figure('Lidar camera position')
        ax_lidar_cam = figure_lidar_cam_pos.add_subplot(111, projection='3d')
        # draw x y z axis that go through the origin
        length = 0.5
        ax_cam.quiver(0, 0, 0, length, 0, 0, color='r')
        ax_cam.quiver(0, 0, 0, 0, length, 0, color='g')
        ax_cam.quiver(0, 0, 0, 0, 0, length, color='b')
        ax_lidar_cam.quiver(0, 0, 0, length, 0, 0, color='r')
        ax_lidar_cam.quiver(0, 0, 0, 0, length, 0, color='g')
        ax_lidar_cam.quiver(0, 0, 0, 0, 0, length, color='b')
        for frame_no, [frame_idx, frame] in enumerate(self.annotation.iterrows()):
            if frame_no == 25:
                break
            print(f'frame {frame_no}/{len(self.annotation)}', end='\r')
            cameraPosition, depth_map, localToWorld, camera_rot_3x3M, depth_cam_intrinsic_3x3M = self.load_lidar(
                frame_no)
            corners = self.get_checkerboard(frame_no)
            # camera4x4M = np.eye(4)
            # camera4x4M[:3, :3] = camera_rot_3x3M
            # camera4x4M[:3, 3] = cameraPosition.reshape((3,))
            translate = np.zeros((4,4))
            translate[2,3] = 2
            translate[1,3] = -0.5
            localToWorld = localToWorld + translate.T
            # todo: draw a 3D line to one of the door key points
            figure_lidar_cam_pos, ax_lidar_cam = draw_camera(localToWorld.T, self.camera_matrix, figure_ax=[figure_lidar_cam_pos,ax_lidar_cam],
                                                    cameraName=frame_no)
            # corners = False
            if corners is False:
                print(f'frame {frame_no} has no checkerboard')
            else:
                localToWorld = self.get_gt_camera_pose(corners)[4]
                figure_gt_cam_pos, ax_cam = draw_camera(localToWorld.T, self.camera_matrix, figure_ax=[figure_gt_cam_pos,ax_cam], cameraName=frame_no)
            self.cameraPositions.append(cameraPosition)
            self.depthMaps.append(depth_map)
            self.localToWorlds.append(localToWorld)
            Lidar_depth, weight = self.extract_kps_depth(frame_no, filter_range=self.filter_range)
            self.Lidar_depths.append(Lidar_depth)
            self.weights.append(weight)
            lineP_3d = self.project_kps(frame_no)
            self.lineP_3ds.append(lineP_3d)
            # method 3
            self.inFrame_measurements.append(measure_obj(lineP_3d, self.config['dist_sequences']))
            self.measurement_weights.append(measure_obj(lineP_3d, self.config['dist_sequences']))
        self.inFrame_measurements = np.array(self.inFrame_measurements)
        self.lineP_3ds = np.array(self.lineP_3ds)
        self.weights = np.array(self.weights)
        plt.show()

