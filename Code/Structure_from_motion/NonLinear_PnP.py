import numpy as np
from scipy.spatial.transform import Rotation 
import scipy.optimize as opt


def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def homogenuous_matrix(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def NonLinearPnP(camera_matrix, point_correspondences, initial_3d_point, initial_rotation, initial_translation):
    """
    camera_matrix : Camera Matrix
    point_correspondences : Point Correspondences
    initial_3d_point : Initial 3D point
    initial_rotation, initial_translation : Relative camera pose - estimated from PnP
    Returns:
        optimized_3d_point : Optimized 3D points
    """

    quaternion = getQuaternion(initial_rotation)
    initial_params = [quaternion[0], quaternion[1], quaternion[2], quaternion[3], initial_translation[0], initial_translation[1], initial_translation[2]]

    optimized_params = opt.least_squares(
        fun=pnp_loss,
        x0=initial_params,
        method="trf",
        args=[initial_3d_point, point_correspondences, camera_matrix]
    )

    optimized_params = optimized_params.x
    optimized_quaternion = optimized_params[:4]
    optimized_translation = optimized_params[4:]
    optimized_rotation = getRotation(optimized_quaternion)

    return optimized_rotation, optimized_translation

def pnp_loss(parameters, initial_3d_point, point_correspondences, camera_matrix):
    quaternion, translation = parameters[:4], parameters[4:].reshape(-1, 1)
    rotation = getRotation(quaternion)
    projection_matrix = ProjectionMatrix(rotation, translation, camera_matrix)

    error = []
    for x, pt in zip(initial_3d_point, point_correspondences):
        p1, p2, p3 = projection_matrix
        p1, p2, p3 = p1.reshape(1, -1), p2.reshape(1, -1), p3.reshape(1, -1)

        x = homogenuous_matrix(x.reshape(1, -1)).reshape(-1, 1)

        u, v = pt[0], pt[1]
        u_proj = np.divide(p1.dot(x), p3.dot(x))
        v_proj = np.divide(p2.dot(x), p3.dot(x))

        e = np.square(v - v_proj) + np.square(u - u_proj)
        error.append(e)

    sum_error = np.mean(np.array(error).squeeze())
    return sum_error


