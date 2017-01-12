import scipy.sparse as sps
import numpy as np
from sksparse.cholmod import cholesky

N = 20
mat = sps.eye(N, format='csc')

matll = cholesky(mat)

vec = np.ones((N, 1))
sol = matll.solve_A(vec)

print '{0} shall be zero'.format(np.linalg.norm(vec - mat*sol))
