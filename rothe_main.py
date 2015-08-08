import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou
import rothe_utils as rtu

dolfin.parameters.linear_algebra_backend = 'uBLAS'

'''
Idea of code

 - define reference FEM spaces `Vref`, `Qref` (fine)
 - compute initial value `vini` on the refspacs and use it as first `vprev`
 - compute `vref`, `pref` on the refspacs using `solve_nse`
 - define `Vplt`, `Qplt` for plotting (rough)
 - time loop:
   - define/get the current spaces `Vcur`, `Qcur`
   - comp `vcur`, `pcur` based on `vprev`
   - comp `and collect norm(vcur - vref)` and `norm(pcur - pref)`
   - plot `vcur`, `pcur` interpolated in the refspacs

Issues:
 - save the `v` variables with boundary conditions, because
   - otherwise `diribcs` have to be saved with the data as well
   - easier for visualization and error estimation
 - save `v` and `p` separately
   - to append the bcs, `v` is separated anyways
   - that is coherent with `solve_nse`
'''

Nref = 3
Nplt = 2

proutdir = 'results/'
ddir = 'data/'

debug = False


def check_the_sim(problem='cylinderwake', index=2, nswtchl=[2], scheme='CR',
                  nu=None, Re=None, Nts=None, paraout=False, t0=None, tE=None,
                  dtstrdct={}, debug=False):

    refvdict, refpdict = gettheref(problem=problem, N=Nref, nu=nu, Re=Re,
                                   t0=t0, tE=tE, Nts=Nts,
                                   scheme=scheme, dtstrdct=dtstrdct,
                                   debug=debug)
    viniv = dou.load_npa(refvdict[t0])
    piniv = dou.load_npa(refpdict[t0])

    vdict, pdict = rtu.rothe_ind2(problem=problem, scheme=scheme, Re=Re, nu=nu,
                                  t0=t0, tE=tE, Nts=Nts,
                                  nswtchl=nswtchl, Nini=Nref,
                                  dtstrdct=dtstrdct)


def gettheref(problem='cylinderwake', N=None, nu=None, Re=None, Nts=None,
              paraout=False, t0=None, tE=None, scheme=None, dtstrdct={},
              debug=False):
    trange = np.linspace(t0, tE, Nts+1)
    refmeshdict = rtu.get_curmeshdict(problem=problem, N=N, nu=nu,
                                      Re=Re, scheme=scheme)

    refvdict, refpdict = snu.\
        solve_nse(trange=trange, clearprvdata=debug, data_prfx=ddir,
                  vfileprfx=proutdir, pfileprfx=proutdir,
                  output_includes_bcs=True,
                  return_dictofvelstrs=True, return_dictofpstrs=True,
                  start_ssstokes=True, **refmeshdict)

    return refvdict, refpdict


if __name__ == '__main__':
    problem = 'cylinderwake'
    scheme = 'CR'
    t0, tE, Nts = 0.0, 0.2, 128
    nswtchl = [3, 2]

    nswtchstr = 'Nswitches' + ''.join(str(e) for e in nswtchl)
    dtstrdct = dict(prefix=ddir+problem+scheme+nswtchstr,
                    method=2, N=None, Nts=Nts, t0=t0, te=tE)
