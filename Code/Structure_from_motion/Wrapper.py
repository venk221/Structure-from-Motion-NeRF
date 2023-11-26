import numpy as np
import matplotlib.pyplot as plt
from FeatureExtraction import *
from Fundamental_Matrix import *
from RANSAC import *
from Essential_Matrix import *
from Inlier_representation import *
from Camera_Pose_Extraction import *
from Linear_Traingulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from Linear_PnP import *
from PnP_Ransac import *
from NonLinear_PnP import *
from Visibility_Matrix import *
from Bundle_Adjustment import *




prev_rot = np.eye(3)
prev_trans =  np.zeros((3,1))
C1 = np.zeros((4, 3, 1))
R1 = np.zeros((4,3,3))
for i in range(4):
    R1[i, :, :] = np.identity(3)
BA = 1
inliers = []
path = '/home/venk/Downloads/SfM&NeRF-P3/Phase1/Data/P3Data'
x_coordinate, y_coordinate, occurance_flag, rgb_vals = features_extraction(path)
filtered_occurance_flag = np.zeros_like(occurance_flag)
fundamentalMatrix = np.empty(shape=(5,5), dtype=object)
C1shape, R1shape = C1.shape, R1.shape
n = 5
for i in range(1,n):
    for j in range(i+1, n+1):
        idx = np.where(occurance_flag[:,i-1] & occurance_flag[:,j-1])
        pts1 = np.hstack((x_coordinate[idx,i-1].reshape((-1,1)), y_coordinate[idx,i-1].reshape((-1,1))))
        pts2 = np.hstack((x_coordinate[idx,j-1].reshape((-1,1)), y_coordinate[idx,j-1].reshape((-1,1))))
        idx = np.array(idx).reshape(np.array(idx).shape[1],1)
        
        if len(idx) > 8:
            F_inliers, inliers_idx = GetInlierRANSANC(pts1,pts2,idx)
            print("In the Images:",i,"and",j,"Found Inliers:", len(inliers_idx),"out of", len(idx) )
            fundamentalMatrix[i-1,j-1] = F_inliers
            filtered_occurance_flag[inliers_idx,j-1] = 1
            filtered_occurance_flag[inliers_idx,i-1] = 1
       
        

print("Corresponding Points extracted from RANSAC")

F12 = fundamentalMatrix[0,1]
E12 = EssentialMatrixFromFundamentalMatrix(F12, K)
C_set, R_set, P_set = ExtractCameraPose(E12, K)

idx = np.where(filtered_occurance_flag[:,0] & filtered_occurance_flag[:,1])
pts1 = np.hstack((x_coordinate[idx,0].reshape((-1,1)), y_coordinate[idx,0].reshape((-1,1))))
pts2 = np.hstack((x_coordinate[idx,1].reshape((-1,1)), y_coordinate[idx,1].reshape((-1,1))))


R1_ = np.identity(3)
C1_ = np.zeros((3,1))
pts3D_4 = []
for i in range(len(C_set)):
    x1 = pts1
    x2 = pts2
    X = LinearTriangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
    X = np.squeeze(X, axis=1)
    pts3D_4.append(X)

pts3D_4 = np.array(pts3D_4)
C_best, R_best, X = DisambiguateCameraPose(C_set,R_set,pts3D_4)

C_best = np.reshape(C_best, (C_best.shape[0],1))
plt.scatter(X[:,0], X[:,2],marker='.')
plt.title("Disambiguated World Points")
plt.xlim([-20,20])
plt.ylim([-25,25])
plt.show()

print("Non-Linear Triangulation")
X_refined = NonLinearTriangulation(K,pts1,pts2,R1_,C1_,R_best,C_best, X)
X_refined = np.reshape(X_refined, (X.shape[0],3))

bestWP = np.reshape(X_refined, (X_refined.shape[0],3))
plt.scatter(bestWP[:,0], bestWP[:,2], marker = '.', cmap = "spring", label = "Non-linear Triangulation")
plt.scatter(X[:,0], X[:,2],  marker = '.', cmap = "cool", label = "Linear Triangulation")
plt.legend()
plt.title("Non-linear vs Linear triangulation")
plt.xlim([-20,20])
plt.ylim([-25,25])
plt.show()

WP_all = np.zeros((x_coordinate.shape[0],3))
cam_indices = np.zeros((x_coordinate.shape[0],1), dtype = int)
X_found = np.zeros((x_coordinate.shape[0],1), dtype = int)

WP_all[idx] = X[:,:3]
X_found[idx] = 1
cam_indices[idx] = 1
X_found[np.where(WP_all[:2]<0)] = 0

C_set = []
R_set = []

C0 = np.zeros(3)
R0 = np.identity(3)
C_set.append(C0)
R_set.append(R0)
C_set.append(C_best)
R_set.append(R_best)


for i in range(2,5):
    print("Adding Image ", i+1)
    feature_idx_i = np.where(X_found[:,0] & filtered_occurance_flag[:,i])
    if len(feature_idx_i[0]) < 8:
        print("Less than 8 points")
        continue

    pts_i = np.hstack((x_coordinate[feature_idx_i, i].reshape(-1,1), y_coordinate[feature_idx_i, i].reshape(-1,1)))
    X = WP_all[feature_idx_i,:].reshape(-1,3)

    R_init, C_init = PnPRANSAC(K,pts_i,X, iter=1000, thresh=5)
    linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)
    
    Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
    non_linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)

    C_set.append(Ci)
    R_set.append(Ri)

    for k in range(0,i):
            idx_X_pts = np.where(filtered_occurance_flag[:,k] & filtered_occurance_flag[:,i])
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)

            if (len(idx_X_pts)<8):
                continue

            x1 = np.hstack((x_coordinate[idx_X_pts,k].reshape(-1,1), y_coordinate[idx_X_pts,k].reshape(-1,1)))
            x2 = np.hstack((x_coordinate[idx_X_pts,i].reshape(-1,1), y_coordinate[idx_X_pts,i].reshape(-1,1)))

            WP_ = LinearTriangulation(K,C_set[k],R_set[k],Ci,Ri,x1,x2)

            WP_ = np.squeeze(WP_, axis=1)
            error_linear = []
            pts1 , pts2 = x1, x2
            WP__ = np.reshape(WP_[:,:3], (WP_.shape[0],3))

            X = NonLinearTriangulation(K,x1,x2,R_set[k],C_set[k],Ri,Ci,WP__)


            bestWP = np.reshape(X, (WP__.shape[0],3))
            plt.scatter(bestWP[:,0], bestWP[:,2], marker = '.', cmap = "spring", label = "Non-linear Triangulation")
            plt.scatter(WP_[:,0], WP_[:,2],  marker = '.', cmap = "cool", label = "Linear Triangulation")
            plt.legend()
            plt.title("Non-linear vs Linear triangulation")
            plt.xlim([-20,20])
            plt.ylim([-25,25])
            plt.show()
            
            X = np.reshape(X, (WP__.shape[0],3))
            WP_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1

            print("New", idx_X_pts[0], "Points Between ", k, "and ",i, "images" )

            WP_indices, visibility_matrix = Visibility_Matrix(X_found,filtered_occurance_flag,nCam=i)
            
            print("Bundle Adjustment:")
            R_set_, C_set_, WP_all = bundle_adjustment(WP_indices, visibility_matrix, WP_all, X_found, x_coordinate, 
                                                       y_coordinate, filtered_occurance_flag, R_set, C_set, K, nCam=i)
            
            for k in range(0,i+1):
                idx_X_pts = np.where(X_found[:,0] & filtered_occurance_flag[:,k])
                x = np.hstack((x_coordinate[idx_X_pts,k].reshape(-1,1), y_coordinate[idx_X_pts,k].reshape(-1,1)))
                X = WP_all[idx_X_pts]
                BundAdj_error = reprojectionErrorPnP(X,x,K,R_set_[k],C_set_[k])

    print(X_found.shape, WP_all.shape)
    X_found[WP_all[:,2]<0] = 0

    feature_idx = np.where(X_found[:,0])
    X = WP_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    fig = plt.figure(figsize = (10,10))
    plt.xlim(-4,6)
    plt.ylim(-2,12)
    plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set_)):
        R1 = get_euler(R_set_[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0],C_set_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')

    plt.show()

    # fig1= plt.figure(figsize= (5,5))
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(x,y,z,color="green")
    # plt.show()