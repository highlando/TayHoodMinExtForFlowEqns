import dolfin
import numpy as np
from time_int_schemes import expand_vp_dolfunc
import dolfin_navier_scipy.problem_setups as dnsps
from prob_defs import FempToProbParams

N, Re, scheme = 2, 50, 'CR'

femp, stokesmatsc, rhsd_vfrc, \
    rhsd_stbc, data_prfx, ddir, proutdir \
    = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                        scheme=scheme)

PrP = FempToProbParams(N, femp=femp, pdof=None)

vp = np.load('data/_m1_N2_nu0.002_Nts512_tol_0.000244140625_t1.0.npy')
v, p = expand_vp_dolfunc(PrP, vp=vp)

dolfin.plot(v)
dolfin.plot(p)
dolfin.interactive(True)
