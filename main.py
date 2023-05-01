from LidarOutput import *
from utils.tools import *

config_file = 'config/1_init_test/Mair.yaml'
config_file = 'config/1_init_test/door.yaml'
config_file = 'config/1_init_test/MEPbox.yaml'
config_file = 'config/2_odometry_check/door.yaml'
# lidar = PhoneLidar(config_file)

# for measurement_no in range(lidar.config['dist_gt'].__len__()):
#     this_measurement = lidar.inFrame_measurements[:,measurement_no].reshape((-1))*1000
#     # this_measurement = [1,2,3,4,10,10]
#     this_gt = lidar.config['dist_gt'][measurement_no]
#     plot_histogram(this_measurement, this_gt, measurement_no)
#     measurements = this_measurement


lidar = PhoneLidarCheckerboardValidate(config_file)

