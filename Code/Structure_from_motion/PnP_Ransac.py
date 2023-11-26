import numpy as np
from Linear_PnP import PnP


def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))   
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def homogenuous_matrix(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = ProjectionMatrix(R,C,K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = homogenuous_matrix(X.reshape(1,-1)).reshape(-1,1) 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error

def PnPError(feature, X, R, C, K):
    u,v = feature
    pts = X.reshape(1,-1)
    X = np.hstack((pts, np.ones((pts.shape[0],1))))
    X = X.reshape(4,1)
    C = C.reshape(-1,1)
    P = ProjectionMatrix(R,C,K)
    p1, p2, p3 = P
    p1, p2, p3 = p1.reshape(1,4), p2.reshape(1,4), p3.reshape(1,4)
    u_proj = np.divide(p1.dot(X), p3.dot(X))
    v_proj = np.divide(p2.dot(X), p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u,v))
    err = np.linalg.norm(x-x_proj)

    return err



def PnPRANSAC(K, x, X, iter=1000, thresh=5):
    """
    This function implements the PnP RANSAC algorithm to estimate the camera pose from a set of 3D-2D correspondences.

    Args:
        K: The camera calibration matrix
        x: The 2D feature points in the image
        X: The 3D points corresponding to the 2D feature points
        iter: The number of RANSAC iter
        thresh: The inlier threshold

    Returns:
        R_best: The estimated rotation matrix
        t_best: The estimated translation vector
    """
    max_inliers = 0
    best_rotation = None
    best_translation = None
    num_rows = X.shape[0]

    for _ in range(iter):
        random_indices = np.random.choice(num_rows, size=6)
        selected_3d_pts = X[random_indices]
        selected_features = x[random_indices]

        rotation, translation = PnP(selected_3d_pts, selected_features, K)

        inliers = find_inliers(x, X, rotation, translation, K, thresh)

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_rotation = rotation
            best_translation = translation

    return best_rotation, best_translation

def find_inliers(x, X, rotation, translation, K, thresh):
    inliers = []

    if rotation is not None:
        for i in range(X.shape[0]):
            feature = x[i]
            threeD_point = X[i]
            error = PnPError(feature, threeD_point, rotation, translation, K)

            if error < thresh:
                inliers.append(i)

    return inliers