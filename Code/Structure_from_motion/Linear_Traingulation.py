import numpy as np

def skewsym(x):
   return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
  """
  Inputs: Intrinsic Matrix, Camera Translation vector (1st camera), Camera Rotation Matrix (1st Camera)
  Camera Translation vector (2nd camera), Camera Rotation Matrix (2nd Camera), point correspondances.
  Output: 3D world points w.r.t camera frame.
  """
  # print(C1.shape, R1.shape, C2.shape, R2.shape, x1.shape, x2.shape)
  ProjectionMatrix = np.zeros((3,4,2))
  # print(C1.shape)
  temp1 = np.append(np.identity(3), -C1.reshape(3, 1), axis=1)
  temp2 = np.array([[1,0,0,C2[0]],[0,1,0,C2[1]],[0,0,1,C2[2]]])
  ProjectionMatrix[:,:,0] = np.matmul(K, np.matmul(R1, temp1))
  ProjectionMatrix[:,:,1] = np.matmul(K, np.matmul(R2, temp2))
  pts1 = np.append(x1, np.reshape(np.ones((x1.shape[0])),(x1.shape[0],1)), axis=1)
  pts2 = np.append(x2, np.reshape(np.ones((x2.shape[0])),(x2.shape[0],1)), axis=1)
  X = []
  for i in range(pts1.shape[0]):
    tem1 = np.matmul(skewsym(pts1[i,:]), ProjectionMatrix[:,:,0])
    tem2 = np.matmul(skewsym(pts2[i,:]), ProjectionMatrix[:,:,1])
    A = np.vstack((tem1, tem2))
    u,s,vt = np.linalg.svd(A)
    x = vt[-1] / vt[-1, -1]
    x = np.reshape(x, (len(x), -1)).transpose()
    X.append(x)
  X = np.array(X)
  return X