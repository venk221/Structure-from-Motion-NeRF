import numpy as np
import scipy.optimize as opt

def NonLinearTriangulation(K, x1, x2, R1, C1, R2, C2, X_init):
  """
  Inputs: Intrinsic Matrix, image points (2D) for both images, disambiguited Rotation matrix 
  and translation vector for both images and X_init = Best Rotation Matrix, Best translation vector
  and Best estimates for 3D world points.
  output: Non linearly (least squares) Optimized 3D world points.
  """
  if x1.shape[0] != X_init.shape[0] or x1.shape[0] != X_init.shape[0]:
    print("shapes are not equal")
  else:
    init = X_init
    init=init.flatten()
    # minimize(init, K, x1, x2, R1, C1, R2, C2)
    optimized_params = opt.least_squares(
    fun=minimize,
    x0=init,
    # method="dogbox",
    args=[K, x1, x2, R1, C1, R2, C2]) 
    return optimized_params.x

def minimize(init, K, x1, x2, R1, C1, R2, C2): 
  temp = init.reshape((-1, 3))  #init.reshape((x1.shape[0],3))
  e1 = 0
  e2 = 0
  X = np.hstack((temp, np.ones((temp.shape[0], 1))))
  temp1 = np.hstack((np.identity(3), -C1.reshape(-1, 1)))
  PM1 = np.matmul(K, np.matmul(R1,temp1))
  temp2 = np.hstack((np.identity((3)),-C2.reshape(-1, 1)))
  PM2 = np.matmul(K, np.matmul(R2,temp2))
  p11, p12, p13 = np.reshape(PM1[0,:],(PM1.shape[1],1)), np.reshape(PM1[1,:],(PM1.shape[1],1)), np.reshape(PM1[2,:],(PM1.shape[1],1))
  p21, p22, p23 =np.reshape(PM2[0,:],(PM2.shape[1],1)), np.reshape(PM2[1,:],(PM2.shape[1],1)), np.reshape(PM2[2,:],(PM2.shape[1],1))
  frac11 = 0
  frac12 = 0
  frac11 = np.divide(np.matmul(p11.transpose(),X.transpose()),np.matmul(p13.transpose(),X.transpose())).transpose()
  frac12 = np.divide(np.matmul(p12.transpose(),X.transpose()),np.matmul(p13.transpose(),X.transpose())).transpose()
  u1, v1 = x1[:,0], x1[:,1]
  term11 = np.square(frac11 - np.reshape(u1, (x1[:,0].shape[0],1))) 
  term12 = np.square(frac12 - np.reshape(v1, (x1[:,1].shape[0],1)))
  e1 = np.sqrt(term11+term12)

  frac21 = 0
  frac22 = 0
  frac21 = np.divide(np.matmul(p21.transpose(),X.transpose()),np.matmul(p23.transpose(),X.transpose())).transpose()
  frac22 = np.divide(np.matmul(p22.transpose(),X.transpose()),np.matmul(p23.transpose(),X.transpose())).transpose()
  u2, v2 = x2[:,0], x2[:,1]
  term21 = np.square(frac21 - np.reshape(u2, (x2[:,0].shape[0],1))) 
  term22 = np.square(frac22 - np.reshape(v2, (x2[:,1].shape[0],1)))
  e2 = np.sqrt(term21+term22)

  error = sum(sum(e1),sum(e2))
  return error