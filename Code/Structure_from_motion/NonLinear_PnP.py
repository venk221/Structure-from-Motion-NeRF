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

# def NonLinearPnP(K, pts, x3D, R0, C0):
#     """    
#     K : Camera Matrix
#     pts1, pts2 : Point Correspondences
#     x3D :  initial 3D point 
#     R2, C2 : relative camera pose - estimated from PnP
#     Returns:
#         x3D : optimized 3D points
#     """

#     Q = getQuaternion(R0)
#     X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 

#     optimized_params = opt.least_squares(
#         fun = PnPLoss,
#         x0=X0,
#         method="trf",
#         args=[x3D, pts, K])
#     X1 = optimized_params.x
#     Q = X1[:4]
#     C = X1[4:]
#     R = getRotation(Q)
#     return R, C

# def PnPLoss(X0, x3D, pts, K):
    
#     Q, C = X0[:4], X0[4:].reshape(-1,1)
#     R = getRotation(Q)
#     P = ProjectionMatrix(R,C,K)
    
#     Error = []
#     for X, pt in zip(x3D, pts):

#         p_1T, p_2T, p_3T = P# rows of P
#         p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)


#         X = homo(X.reshape(1,-1)).reshape(-1,1) 
#         ## reprojection error for reference camera points 
#         u, v = pt[0], pt[1]
#         u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
#         v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

#         E = np.square(v - v_proj) + np.square(u - u_proj)

#         Error.append(E)

#     sumError = np.mean(np.array(Error).squeeze())
#     return sumError


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


