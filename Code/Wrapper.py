import numpy as np
import sys
import glob
import cv2
import math 
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as opt


from LoadData import *
from GetInliersRANSAC import *

from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *

from LinearTriangulation import *
from Disambiguate import *
from NonLinearTriangulation import *

from error_comparision import *




K = np.array([[531.122155322710, 0, 407.192550839899],[0, 531.541737503901, 313.308715048366],[0, 0 ,1]])


def main():

    matches1and2 = findMatches(1,2)
    bestF, inliers = GetInlierRANSANC(matches1and2[:,4:6], matches1and2[:,7:9])


    inliers = np.array(inliers)
    inliers1and2 = matches1and2[inliers]
    # print(inliers1and2.shape)
    inliers1and2 = np.reshape(inliers1and2, (inliers1and2.shape[0],9))

    E = EssentialMatrixFromFundamentalMatrix(bestF, K)

    C, R, P = ExtractCameraPose(E, K)


    print('Doing Linear Triangulation .....')
    X_set = []
    for n in range(0, 4):
        X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3),
                                    C[n,:].T, R[n,:,:], np.float32(inliers1and2[:,4:6]),
                                    np.float32(inliers1and2[:,7:9]))
        X_set.append(X1)


    X_set = np.array(X_set)
    # print(X_set.shape)
    X_set = np.reshape(X_set, (4,X_set.shape[1],4))

    x1, z1 = X_set[0,:,0], X_set[0,:,2]
    x2, z2 = X_set[1,:,0], X_set[1,:,2]
    x3, z3 = X_set[2,:,0], X_set[2,:,2]
    x4, z4 = X_set[3,:,0], X_set[3,:,2]

    _=plt.figure()

    plt.scatter(x1,z1, cmap = 'bone',label='Pose1')
    plt.scatter(x2,z2, cmap = 'spring',label='Pose2')
    plt.scatter(x3,z3, cmap = 'cool',label='Pose3')
    plt.scatter(x4,z4, cmap = 'pink',label='Pose4')
    plt.title("World Points with 4 camera poses")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig('WP_4.png')
    # plt.show()

    Cbest, Rbest, Xbest = DisambiguateCameraPose(C, R, X_set)
    Cbest = np.reshape(Cbest, (Cbest.shape[0],1))


    _=plt.figure()
    plt.scatter(Xbest[:,0], Xbest[:,2])
    plt.title("Disambiguated World Points")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig('WP_1.png')

    print('Doing Non Linear Triangulation .....')
    X = NonLinearTriangulation(K, inliers1and2[:,4:6], inliers1and2[:,7:9], np.eye(3), np.zeros((3,1)), Rbest, Cbest, Xbest)

    bestWP = np.reshape(X, (inliers1and2.shape[0],3))
    _=plt.figure()
    plt.scatter(bestWP[:,0], bestWP[:,2], cmap = "spring", label = "Non-linear Triangulation")
    plt.scatter(Xbest[:,0], Xbest[:,2], cmap = "cool", label = "Linear Triangulation")
    plt.legend()
    plt.title("Non-linear vs Linear triangulation")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig('NonLinearVsLinear.png')


    ## Non Linear Vs Linear Error Calculation
    ErrorComparison(K,Rbest,Cbest,bestWP,Xbest,inliers1and2)









    
main()



