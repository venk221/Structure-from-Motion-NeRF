import numpy as np

def ExtractCameraPose(E, K):
  """
  Inputs: Essential Matrix and Intrinsic Matrix
  Output: 4 Poses which needs disambuigation 
  (4 Rotation Matrices and 4 Translation vectors)
  """
  w = np.array([[0,-1,0],[1,0,0],[0,0,1]])
  u, s, vt = np.linalg.svd(E)
  P = []
  R = []
  C = []
  C.append(u[:,2])
  C.append(-u[:,2])
  C.append(u[:,2])
  C.append(-u[:,2])
  C = np.array(C)
  # print("trey",C)
  R.append(np.matmul(u,np.matmul(w,vt)))
  R.append(np.matmul(u,np.matmul(w,vt)))
  R.append(np.matmul(u,np.matmul(w.transpose(),vt)))
  R.append(np.matmul(u,np.matmul(w.transpose(),vt)))
  R = np.array(R)
  # print("treyR",R)
  P.append(np.matmul(np.matmul(K,R[0,:,:]),np.array([[1,0,0,C[0,0]],[0,1,0,C[0,1]],[0,0,1,C[0,2]]])))
  P.append(np.matmul(np.matmul(K,R[1,:,:]),np.array([[1,0,0,C[1,0]],[0,1,0,C[1,1]],[0,0,1,C[1,2]]])))
  P.append(np.matmul(np.matmul(K,R[2,:,:]),np.array([[1,0,0,C[2,0]],[0,1,0,C[2,1]],[0,0,1,C[2,2]]])))  
  P.append(np.matmul(np.matmul(K,R[3,:,:]),np.array([[1,0,0,C[3,0]],[0,1,0,C[3,1]],[0,0,1,C[3,2]]])))
  P = np.array(P)

  Rnew = []
  Cnew = []

  epsilon=10e-6
  for r,c in zip(R,C):
    if (np.linalg.det(r) <= float(1)+epsilon) or (np.linalg.det(r)>=float(1)-epsilon) :
      Rnew.append(r)
      Cnew.append(c)
    elif np.linalg.det(r) < 0:
      Rnew.append(-r)
      Cnew.append(-c)

    else:
      print('Exit')
      print(np.linalg.det(r))
  Rnew = np.array(Rnew)
  Cnew = np.array(Cnew)
  # print("test",Rnew, Cnew)   
  return Cnew, Rnew, P