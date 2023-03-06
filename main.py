from LidarOutput import PhoneLidar
from utils.tools import *

config_file = 'config/Mair.yaml'
# config_file = 'config/door.yaml'
# config_file = 'config/MEPbox.yaml'

lidar = PhoneLidar(config_file)

for measurement_no in range(lidar.config['dist_gt'].__len__()):
    this_measurement = lidar.inFrame_measurements[:,measurement_no].reshape((-1))*1000
    # this_measurement = [1,2,3,4,10,10]
    this_gt = lidar.config['dist_gt'][measurement_no]
    plot_histogram(this_measurement, this_gt, measurement_no)
    measurements = this_measurement