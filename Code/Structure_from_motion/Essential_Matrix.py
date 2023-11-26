import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
  """Inputs: Fundamental Matrix, Intrinsic Matrix
     output: Essential Matrix
  """
  E = np.dot(K.T, np.dot(F, K))
  u, s, vt = np.linalg.svd(E)
  E = np.dot(u, np.dot(np.diag([1, 1, 0]), vt))
  return E