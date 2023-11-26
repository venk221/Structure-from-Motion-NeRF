import numpy as np

def EstimateFundamentalMatrix(a, b):
  """
  Input: Point Correspondances in image 1 (a) and image 2 (b). 
  Output: A fundamental matrix (Needs optimization for all point correspondances)
  """
  # print(np.sum(a[:,0]))
  n = a.shape[0]
  # print(a,b)
  """Normalization - Eight Point Algorithm"""
  #Calculating means
  ua = (1/n)*(np.sum(a[:,0]))
  va = (1/n)*(np.sum(a[:,1]))
  ub = (1/n)*(np.sum(b[:,0]))
  vb = (1/n)*(np.sum(b[:,1]))
  ua_dash, va_dash = a[:,0] - ua, a[:,1] - va
  ub_dash, vb_dash = b[:,0] - ub, b[:,1] - vb 
  #Finding scale = https://en.wikipedia.org/wiki/Eight-point_algorithm#:~:text=The%20eight%2Dpoint%20algorithm%20is,case%20of%20the%20essential%20matrix.
  sa = n / np.sum(((ua_dash)**2 + (va_dash)**2)**(0.5))  #(2**(0.5))/(((1/n)*(np.mean((ua_dash**2)+(va_dash**2))))**(0.5))
  sb = n / np.sum(((ub_dash)**2 + (vb_dash)**2)**(0.5))  #(2**(0.5))/(((1/n)*(np.mean((ub_dash**2)+(vb_dash**2))))**(0.5)) 
  scaleA = np.array([[sa, 0, 0],[0, sa, 0],[0, 0, 1]])
  scaleB = np.array([[sb, 0, 0],[0, sb, 0],[0, 0, 1]])
  translationA = np.array([[1, 0, -ua],[0, 1, -va],[0, 0 , 1]])
  translationB = np.array([[1, 0, -ub],[0, 1, -vb],[0, 0 , 1]])
  transformation_A_2D = np.matmul(scaleA, translationA)
  transformation_B_2D = np.matmul(scaleB, translationB) 
  aPrev = np.column_stack((a, np.ones(len(a))))
  bPrev = np.column_stack((b, np.ones(len(b))))
  a_normalized = np.matmul(transformation_A_2D, aPrev.transpose())
  b_normalized = np.matmul(transformation_B_2D, bPrev.transpose())
  anew, bnew = a_normalized.transpose(), b_normalized.transpose()
  A = []
  for i in range(0, anew.shape[0]): #anew.shape[0]
    x1, y1 = anew[i,0], anew[i,1]
    x2, y2 = bnew[i,0], bnew[i,1]
    A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])  #[x1*x2, y2*x1, x1, x2*y1, y1*y2, y1, x2, y2, 1] [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
  A = np.array(A)
  _, s, v = np.linalg.svd(A, full_matrices=True)
  v = v.transpose()
  F = v[:,-1]
  F = np.reshape(F, (3, 3))
  """SVD Cleanup"""
  F = np.dot(transformation_B_2D.transpose(), np.dot(F, transformation_A_2D))
  u, s1, vt = np.linalg.svd(F)
  s = np.diag(s1)
  s[2,2] = 0
  F = np.dot(u, np.dot(s, vt))
  F = F/F[2,2]
  return F