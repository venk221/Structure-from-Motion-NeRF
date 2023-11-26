import numpy as np
from Fundamental_Matrix import *
import random

def error(x1,x2,f):
  x1 = np.append(x1,1) #np.reshape(x1,(1,2))
  x1 = np.reshape(x1,(3,1))
  x2 = np.append(x2,1)
  x2 = np.reshape(x2, (1,3))
  # print(f"bhok = {x1, x1.shape}, x2={x2, x2.shape}, f={f.shape}")
  err = np.dot(x2, np.dot(f, x1))
  return np.abs(err)
  
def GetInlierRANSANC(a,b, idx):
  """
  Input: Point Correspondances in image 1 (a) and image 2 (b)
  Output: A fundamental matrix optimised iteratively (using Ransac)
  """
  num_inliers = 0
  num_indexes = 0
  iterations = 2000
  threshold = 0.005
  ffinal = 0
  final_inliers = []
  max = a.shape[0]
  # print(a[0,:].shape)
  for i in range(iterations):
      indices = []  
    # if i == 99:
      for i in range(0, 8):
        indices.append(random.randint(0, max-1))
      x1 = a[indices,:]
      x2 = b[indices,:]
      ftemp = EstimateFundamentalMatrix(x1,x2)
      inliers = []
      indexes = []
      # if (len(a) > 8) and (len(b) > 8):
      for j in range(len(a)):
        # if j == 1:
        err = error(a[j,:], b[j,:], ftemp)
        # print(err)
        if err < threshold:
                # print("kaka")
                inliers.append([j])
         
      if len(inliers) > num_inliers:
          num_inliers = len(inliers)
          final_inliers = inliers

  # print("index",np.array(inlier_indexes).shape)
  # print("inlier",np.array(final_inliers).shape)
  final_inliers = np.array(final_inliers)
  pointsWithInliersA = a[final_inliers]
  pointsWithInliersB = b[final_inliers]
  pointsWithInliersA = np.reshape(pointsWithInliersA, (pointsWithInliersA.shape[0],2))
  pointsWithInliersB = np.reshape(pointsWithInliersB, (pointsWithInliersB.shape[0],2))
  # print(pointsWithInliersB) 
  ffinal = EstimateFundamentalMatrix(pointsWithInliersA, pointsWithInliersB)
  # print("Best fundamental matrix \n",ffinal, "\nwith no of inliers=", num_inliers)
  return ffinal, final_inliers