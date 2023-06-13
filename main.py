from LidarOutput import *
from utils.tools import *
import c3d

config_file = 'config/1_init_test/Mair.yaml'
config_file = 'config/1_init_test/door.yaml'
config_file = 'config/1_init_test/MEPbox.yaml'
config_file = 'config/2_odometry_check/door.yaml'
config_file = 'config/2_odometry_check/door_w_stabilizer.yaml'
# lidar = PhoneLidar(config_file)

# for measurement_no in range(lidar.config['dist_gt'].__len__()):
#     this_measurement = lidar.inFrame_measurements[:,measurement_no].reshape((-1))*1000
#     # this_measurement = [1,2,3,4,10,10]
#     this_gt = lidar.config['dist_gt'][measurement_no]
#     plot_histogram(this_measurement, this_gt, measurement_no)
#     measurements = this_measurement


lidar = PhoneLidarCheckerboardValidate(config_file)



# depth_file = r'Y:\phone_Lidar\data\odometry_check\2023-05-25_045330\data\1735852.842106041_1.json'
# rgb_file = depth_file.replace('json', 'jpeg')
# depth_points = np.array([[3127,2258],[3000,2278],[2882,2272],[2779,2258],[2639,2235]])
# rgb = cv2.imread(rgb_file)
# # show
# figure1 = plt.figure(1)
# ax1 = figure1.add_subplot(111)
# ax1.imshow(rgb)
# for i in range(depth_points.shape[0]):
#     y,x = int(np.rint(depth_points[i][1])), int(np.rint(depth_points[i][0])) # todo: check main code for order of x and y
#     ax1.scatter(x, y, s=10, c='r', marker='o')
#
# with open(depth_file, 'r') as f:
#     depth_data = json.load(f)
# depth_map = depth_data['depthMap']
# depth_map = np.array(depth_map)
# figure2 = plt.figure(2)
# ax2 = figure2.add_subplot(111)
# ax2.imshow(depth_map)
# depth_img_scale = 256 / 5760
# for i in range(depth_points.shape[0]):
#     y,x = int(np.rint(depth_points[i][1] * depth_img_scale)), int(np.rint(depth_points[i][0] * depth_img_scale)) # todo: check main code for order of x and y
#     ax2.scatter(x, y, s=10, c='r', marker='o')
#     depth = depth_map[y,x]
#     # do average of nearby points
#     b = 2
#     depth = np.mean(depth_map[y-b:y+b,x-b:x+b])
#     # format to 2 decimal places
#     ax2.text(x, y-i*5-5, str('%.2f' % depth), color='black', fontsize=8)
#
#
# plt.show()