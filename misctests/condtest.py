import numpy as np
import numpy.linalg as npla

Nl = [100, 200, 400]

for N in Nl:
    h = 1./N
    Bts = np.diag([1./N]*(N-1), 1)
    Bt = h*np.eye(N)
    # print Bt + Bts
    # print npla.cond(Bt)
    print N, npla.cond(Bt + Bts)

# Bt[0, -1] = h
# print npla.cond(Bt)
