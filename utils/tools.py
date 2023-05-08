import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def dist_penalty(pt1, pt2, threshold):
    return np.sqrt(np.sum((pt1 - pt2) ** 2)) / threshold

def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    https://silo.tips/download/least-squares-intersection-of-lines

    function from: https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
    """
    # generate all line direction vectors
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p


def pts_center_ransac(points, num_iterations=3000, err_threshold=1, weights=None, simple=False):
    # mediumn err_threshold=0.2
    # large err_threshold=0.5
    # small err_threshold=0.1
    best_center = None  # Store the best estimate of the center
    best_inliner_no = 0  # Store the lowest error seen so far
    best_inliner_list = []
    best_inliner_weight = []
    best_loss = np.inf  # Store the lowest error seen so far
    for _ in range(num_iterations):  # Iterate for a given number of iterations
        inliner_list = []
        inliner_weight = []
        loss = 0
        p1, p2, p3 = points[np.random.choice(points.shape[0], 3, replace=False)]  # Randomly choose 3 points
        center = (p1 + p2 + p3) / 3  # Compute the center as the average of the 3 points

        for i, point in enumerate(points):
            point_dist = dist(center, point)
            if point_dist < err_threshold:
                inliner_list.append(point)
                if weights is None:
                    inliner_weight.append(1)
                    loss += (point_dist/err_threshold)**2
                else:
                    if simple:
                        inliner_weight.append(weights[i])
                        loss += 0
                    else:
                        inliner_weight.append(weights[i])
                        loss += weights[i] * np.exp(point_dist/err_threshold)/np.exp(1)
                        # todo: bell curve, exp(x^2/xx^2)
                        # loss += weights[i] * (point_dist/err_threshold)
            else:
                loss += 1

        if loss < best_loss:
            best_center = center
            best_loss = loss
            best_inliner_list = inliner_list
            best_inliner_no = len(inliner_list)
            best_inliner_weight = inliner_weight

    best_inliner_list = np.array(best_inliner_list)
    best_inliner_weight = np.array([best_inliner_weight,best_inliner_weight,best_inliner_weight]).T
    best_center = np.average(best_inliner_list, axis=0, weights=best_inliner_weight)  ## weighted point center
    print('best_inliner_no: ', best_inliner_no)
    return best_center  # Return the center with the lowest error


def pts_center_ransac_old(points, num_iterations=3000, weights=None):
    best_center = None  # Store the best estimate of the center
    lowest_error = float("inf")  # Store the lowest error seen so far
    for _ in range(num_iterations):  # Iterate for a given number of iterations
        p1, p2, p3 = points[np.random.choice(points.shape[0], 3, replace=False)]  # Randomly choose 3 points
        center = (p1 + p2 + p3) / 3  # Compute the center as the average of the 3 points
        if weights is None:
            error = sum([dist(p, center) for p in points])  # Compute the error as the sum of the distances from the center to each point
        else:
            error = sum([weights[i] * dist(p, center) for i, p in enumerate(points)])  # Compute the error as the sum of the distances from the center to each point
        if error < lowest_error:  # If the error is lower than the current lowest error, update the best center
            lowest_error = error
            best_center = center
    return best_center  # Return the center with the lowest error


def plot_3D_points(points, ax):
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10, c='r')


def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (rad)
        order = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1) # * np.pi / 180
    s1 = np.sin(theta1) # * np.pi / 180
    c2 = np.cos(theta2) # * np.pi / 180
    s2 = np.sin(theta2) # * np.pi / 180
    c3 = np.cos(theta3) # * np.pi / 180
    s3 = np.sin(theta3) # * np.pi / 180

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix


def draw_camera(cameraTransform4x4, cameraIntrinsic3x3, resolution = (5760,4320), cameraSize=0.05, cameraColor='r', cameraName='cam', figure_ax = None):
    """
    Draw camera in the scene
    :param cameraTransform4x4: camera transform matrix
    :param cameraIntrinsic3x3: camera intrinsic matrix
    :param cameraSize: camera size
    :param cameraColor: camera color
    :param cameraName: camera name
    :return:
    """

    # get camera transform
    cameraPosition = cameraTransform4x4[:3, 3]

    # four edge points
    basePts_2D = np.array([[0, 0],
                           [0,resolution[1]],
                           [resolution[0],resolution[1]],
                           [resolution[0],0]], dtype=np.float32)

    # project to 3D
    localPoint = (np.linalg.pinv(cameraIntrinsic3x3) @
                  np.hstack(
                      (basePts_2D, np.ones((4, 1)))
                  ).T
                  ).T*cameraSize
    basePts_3D = (cameraTransform4x4 @
                np.hstack(
                    (
                        localPoint, np.ones((4, 1))
                    )
                ).T
                ).T
    # homogeneous to cartesian
    basePts_3D = basePts_3D[:, :3] / basePts_3D[:, 3:]
    # add uppper limit of alpha to be 1
    alpha = (cameraName % 30)/30*0.5 + 0.3

    if figure_ax is None:
        figure = plt.figure(cameraName)
        ax = figure.add_subplot(111, projection='3d')
    else:
        figure, ax = figure_ax
    text_offset = [0.02,0.02,0.05]

    ax.plot3D([basePts_3D[0, 0], basePts_3D[1, 0]],
              [basePts_3D[0, 1], basePts_3D[1, 1]],
              [basePts_3D[0, 2], basePts_3D[1, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([basePts_3D[2, 0], basePts_3D[1, 0]],
              [basePts_3D[2, 1], basePts_3D[1, 1]],
              [basePts_3D[2, 2], basePts_3D[1, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([basePts_3D[2, 0], basePts_3D[3, 0]],
              [basePts_3D[2, 1], basePts_3D[3, 1]],
              [basePts_3D[2, 2], basePts_3D[3, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([basePts_3D[0, 0], basePts_3D[3, 0]],
              [basePts_3D[0, 1], basePts_3D[3, 1]],
              [basePts_3D[0, 2], basePts_3D[3, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([basePts_3D[0, 0], basePts_3D[2, 0]],
              [basePts_3D[0, 1], basePts_3D[2, 1]],
              [basePts_3D[0, 2], basePts_3D[2, 2]], color=cameraColor,alpha = 0.3)
    ax.text(cameraPosition[0] + text_offset[0], cameraPosition[1] + text_offset[1], cameraPosition[2] + text_offset[2], cameraName, color='b', fontsize=8)
    ax.scatter(cameraPosition[0], cameraPosition[1], cameraPosition[2], color='b',alpha = alpha, s = 20)

    # ax.scatter(basePts_3D[1, 0], basePts_3D[1, 1], basePts_3D[1, 2],color='b',alpha = alpha, s = 20)
    ax.plot3D([cameraPosition[0], basePts_3D[0, 0]],
              [cameraPosition[1], basePts_3D[0, 1]],
              [cameraPosition[2], basePts_3D[0, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([cameraPosition[0], basePts_3D[1, 0]],
              [cameraPosition[1], basePts_3D[1, 1]],
              [cameraPosition[2], basePts_3D[1, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([cameraPosition[0], basePts_3D[2, 0]],
              [cameraPosition[1], basePts_3D[2, 1]],
              [cameraPosition[2], basePts_3D[2, 2]], color=cameraColor,alpha = 0.3)
    ax.plot3D([cameraPosition[0], basePts_3D[3, 0]],
              [cameraPosition[1], basePts_3D[3, 1]],
              [cameraPosition[2], basePts_3D[3, 2]], color=cameraColor,alpha = 0.3)

    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')

    # # same scale for all axis
    scale = 1.5
    ax.set_xlim3d(-scale/2, scale/2)
    ax.set_ylim3d(-scale/2, scale/2)
    ax.set_zlim3d(-0, scale)

    # plt.show()
    return figure, ax


def dist(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


def measure_obj(kpts, door_sequences):
    door_dists = np.zeros((len(door_sequences), 1))
    for seq_id, seq in enumerate(door_sequences):
        door_dists[seq_id] = dist(kpts[seq[0]], kpts[seq[1]])
    return door_dists


def measure_obj_weight(weights, door_sequences):
    measurement_weights = np.zeros((len(door_sequences), 1))
    for seq_id, seq in enumerate(door_sequences):
        measurement_weights[seq_id] = weights[seq[0]] * weights[seq[1]]
    return measurement_weights


def compare_gt(door_dists, gt_door_dists):
    door_dists = np.array(door_dists)
    gt_door_dists = np.array(gt_door_dists)
    diff = np.abs(door_dists - gt_door_dists)
    return diff


def plot_histogram(measurements, gt_measurements, title='Histogram'):
    """
    plot a histogram of measurements, and a vertical line for the ground truth, mean, and median
    :param measurements:
    :param gt_measurements:
    :param title:
    :return:
    """
    fig, ax = plt.subplots()
    measurements = measurements[~np.isnan(measurements)]
    ax.hist(measurements, bins=20)
    ax.axvline(np.nanmean(measurements), color='r', linestyle='dashed', linewidth=1, label='mean')
    ax.axvline(np.nanmedian(measurements), color='g', linestyle='dashed', linewidth=1, label='median')
    ax.axvline(gt_measurements, color='black', linestyle='dashed', linewidth=3, label='gt')
    ax.legend()
    plt.title(title)
    plt.show()
    return fig, ax



