import numpy as np
import scipy.linalg as la

A = np.array([[1.0,2.0,3.0,4.0],
              [2.5,2.6,3.7,4.8],
              [3.1,3.9,3.3,3.4],
              [4.1,4.2,4.0,4.4],], dtype=float)
print("A =", A)
print("inv =", np.linalg.inv(A))
print("det =", np.linalg.det(A))

lu, piv = la.lu_factor(A)
print("lu =", lu)
print("piv =", piv)