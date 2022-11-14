import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.

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


def draw_camera(cameraTransform4x4, cameraIntrinsic3x3, resolution = (5760,4320), cameraSize=0.1, cameraColor=(0, 255, 0), cameraName='cam',figure = None):
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
    cameraPosition = cameraTransform4x4[3, :3] # Fixme: curreent local to world

    # four edge points
    basePts_2D = np.array([[0, 0],[0,resolution[1]],[resolution[0],resolution[1]],[resolution[0],0]], dtype=np.float32)

    # project to 3D
    localPoint = (np.linalg.pinv(cameraIntrinsic3x3).T @
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

    if figure is None:
        figure = plt.figure(cameraName)
    ax = figure.gca(projection='3d')
    ax.plot3D([basePts_3D[0, 0], basePts_3D[1, 0]],
              [basePts_3D[0, 1], basePts_3D[1, 1]],
              [basePts_3D[0, 2], basePts_3D[1, 2]], color=cameraColor)
    ax.plot3D([basePts_3D[2, 0], basePts_3D[1, 0]],
              [basePts_3D[2, 1], basePts_3D[1, 1]],
              [basePts_3D[2, 2], basePts_3D[1, 2]], color=cameraColor)
    ax.plot3D([basePts_3D[2, 0], basePts_3D[3, 0]],
              [basePts_3D[2, 1], basePts_3D[3, 1]],
              [basePts_3D[2, 2], basePts_3D[3, 2]], color=cameraColor)
    ax.plot3D([basePts_3D[0, 0], basePts_3D[3, 0]],
              [basePts_3D[0, 1], basePts_3D[3, 1]],
              [basePts_3D[0, 2], basePts_3D[3, 2]], color=cameraColor)
    ax.plot3D([basePts_3D[0, 0], basePts_3D[2, 0]],
              [basePts_3D[0, 1], basePts_3D[2, 1]],
              [basePts_3D[0, 2], basePts_3D[2, 2]], color=cameraColor)
    ax.text(basePts_3D[0, 0], basePts_3D[0, 1], basePts_3D[0, 2], cameraName, color='b')
    ax.scatter(cameraPosition[0], cameraPosition[1], cameraPosition[2], color='b')

    ax.plot3D([cameraPosition[0], basePts_3D[0, 0]],
              [cameraPosition[1], basePts_3D[0, 1]],
              [cameraPosition[2], basePts_3D[0, 2]], color=cameraColor)
    ax.plot3D([cameraPosition[0], basePts_3D[1, 0]],
              [cameraPosition[1], basePts_3D[1, 1]],
              [cameraPosition[2], basePts_3D[1, 2]], color=cameraColor)
    ax.plot3D([cameraPosition[0], basePts_3D[2, 0]],
              [cameraPosition[1], basePts_3D[2, 1]],
              [cameraPosition[2], basePts_3D[2, 2]], color=cameraColor)
    ax.plot3D([cameraPosition[0], basePts_3D[3, 0]],
              [cameraPosition[1], basePts_3D[3, 1]],
              [cameraPosition[2], basePts_3D[3, 2]], color=cameraColor)

    return figure

