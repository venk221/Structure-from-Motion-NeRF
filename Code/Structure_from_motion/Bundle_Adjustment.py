import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from Visibility_Matrix import *

# Source: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

# Function to get rotation matrix from quaternion or rotation vector
def get_rotation(Q, type_='q'):
    if type_ == 'q':
        return Rotation.from_quat(Q).as_matrix()
    elif type_ == 'e':
        return Rotation.from_rotvec(Q).as_matrix()

# Function to get Euler angles from rotation matrix
def get_euler(R2):
    return Rotation.from_matrix(R2).as_rotvec()

# Function to get 2D points from feature x and feature y
def get_2d_points(X_index, visibility_matrix, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visibility_matrix.shape
    for i in range(h):
        for j in range(w):
            if visibility_matrix[i, j] == 1:
                pt = np.hstack((visible_feature_x[i, j], visible_feature_y[i, j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)

# Function to get camera and point indices from visibility matrix
def get_camera_point_indices(visibility_matrix):
    camera_indices, point_indices = [], []
    h, w = visibility_matrix.shape
    for i in range(h):
        for j in range(w):
            if visibility_matrix[i, j] == 1:
                camera_indices.append(j)
                point_indices.append(i)
    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

# Function to create sparsity matrix for bundle adjustment
def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    number_of_cam = nCam + 1
    X_index, visibility_matrix = Visibility_Matrix(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visibility_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(n_observations)
    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam) * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A

# Function to project 3D points onto 2D using camera parameters
def project(points_3d, camera_params, K):
    def project_point_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3, 1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = get_rotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3, 1)
        pt3D = points_3d[i]
        pt_proj = project_point_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)
    return np.array(x_proj)

# Function to rotate points by given rotation vectors
def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

# Function to compute residuals for bundle adjustment
def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    return error_vec

# Main function for bundle adjustment
def bundle_adjustment(X_index, visibility_matrix, X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set,
                      C_set, K, nCam):
    points_3d = X_all[X_index]
    points_2d = get_2d_points(X_index, visibility_matrix, feature_x, feature_y)

    # RC = np.array([np.hstack((get_euler(R_set[i]), C_set[i])) for i in range(nCam + 1)])
    RC = np.array([np.hstack((get_euler(R_set[i])[:, np.newaxis].flatten(), C_set[i][:, np.newaxis].flatten())) for i in range(nCam + 1)])

    x0 = np.hstack((RC.ravel(), points_3d.ravel()))
    n_pts = points_3d.shape[0]

    camera_indices, points_indices = get_camera_point_indices(visibility_matrix)

    A = bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_pts, camera_indices, points_indices, points_2d, K))

    t1 = time.time()

    x1 = res.x
    no_of_cams = nCam + 1
    optim_cam_param = x1[:no_of_cams * 6].reshape((no_of_cams, 6))
    optim_pts_3d = x1[no_of_cams * 6:].reshape((n_pts, 3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_pts_3d

    optim_C_set, optim_R_set = [], []
    for i in range(len(optim_cam_param)):
        R = get_rotation(optim_cam_param[i, :3], 'e')
        C = optim_cam_param[i, 3:].reshape(3, 1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all
