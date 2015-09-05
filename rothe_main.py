import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import rothe_utils as rtu
from time_int_schemes import get_dtstr

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

debug = True
debug = False


def check_the_sim(problem='cylinderwake', index=2, nswtchsl=[2], scheme='CR',
                  nu=None, Re=None, Nts=None, paraout=False, t0=None, tE=None,
                  dtstrdct={}, debug=False):

    trange = np.linspace(t0, tE, Nts+1)
    refdtstrdct = dict(prefix=ddir+problem+scheme,
                       method=2, N=None, Nts=Nts, t0=t0, te=tE)
    refdatstr = get_dtstr(t=None, **refdtstrdct)
    compargs = dict(problem=problem, N=Nref, nu=nu, Re=Re, trange=trange,
                    scheme=scheme, data_prefix=refdatstr, debug=debug)
    refvdict, refpdict = dou.load_or_comp(filestr=[refdatstr+'_refvdict',
                                                   refdatstr+'_refpdict'],
                                          comprtn=gettheref,
                                          comprtnargs=compargs,
                                          debug=debug, itsadict=True)

    viniv = dou.load_npa(refvdict['{0}'.format(t0)])
    piniv = dou.load_npa(refpdict['{0}'.format(t0)])

    # set up the list of the mesh parameters at every time step
    swl = Nts/len(nswtchsl)
    Nll = np.array([nswtchsl[0]]*(Nts+1))
    for k, N in enumerate(nswtchsl[1:]):
        Nll[(k+1)*swl+1:] = N
    Nll[0] = Nref
    Nll = Nll.tolist()

    vdict, pdict = rtu.rothe_ind2(problem=problem, scheme=scheme, Re=Re, nu=nu,
                                  t0=t0, tE=tE, Nts=Nts,
                                  viniv=viniv, piniv=piniv, Nini=Nref,
                                  Nll=Nll, dtstrdct=dtstrdct)

    compvperrargs = dict(problem=problem, scheme=scheme,
                         trange=trange.tolist(),
                         rvd=refvdict, rpd=refpdict, Nref=Nref,
                         cvd=vdict, cpd=pdict, Nll=Nll)

    dtstrdct.update(t=None)
    cdatstr = get_dtstr(**dtstrdct)
    errdict = dou.load_or_comp(filestr=[cdatstr+'_vperrdict'],
                               comprtn=compvperr, debug=debug,
                               comprtnargs=compvperrargs, itsadict=True)

    rtu.plottimeerrs(trange=trange, perrl=[errdict['perrl']],
                     verrl=[errdict['verrl']], showplot=True)


def compvperr(problem=None, scheme=None, trange=None,
              rvd=None, rpd=None, Nref=None,
              cvd=None, cpd=None, Nll=None):
    verrl, perrl = [], []

    rmd = rtu.get_curmeshdict(problem=problem, N=Nref,
                              scheme=scheme, onlymesh=True)
    for tk, t in enumerate(trange):
        vref = dou.load_npa(rvd['{0}'.format(t)])
        pref = dou.load_npa(rpd['{0}'.format(t)])
        vreff, preff = dts.\
            expand_vp_dolfunc(vc=vref, V=rmd['V'], pc=pref, Q=rmd['Q'])

        cmd = rtu.get_curmeshdict(problem=problem, N=Nll[tk],
                                  scheme=scheme, onlymesh=True)
        try:
            vcur = dou.load_npa(cvd[t])
            pcur = dou.load_npa(cpd[t])
        except KeyError:
            vcur = dou.load_npa(cvd['{0}'.format(t)])
            pcur = dou.load_npa(cpd['{0}'.format(t)])

        raise Warning('TODO: debug')
        vcurf, pcurf = dts.\
            expand_vp_dolfunc(vc=vcur, V=cmd['V'], pc=pref, Q=rmd['Q'])
        # vcurf, pcurf = dts.\
        #     expand_vp_dolfunc(vc=vref, V=cmd['V'], pc=pcur, Q=cmd['Q'])
        verrl.append(dolfin.errornorm(vreff, vcurf))
        perrl.append(dolfin.errornorm(preff, pcurf))

    return dict(verrl=verrl, perrl=perrl, trange=trange, Nll=Nll, Nref=Nref)


def gettheref(problem='cylinderwake', N=None, nu=None, Re=None, Nts=None,
              paraout=False, trange=None, scheme=None, data_prefix='',
              debug=False):
    refmeshdict = rtu.get_curmeshdict(problem=problem, N=N, nu=nu,
                                      Re=Re, scheme=scheme)

    refvdict, refpdict = snu.\
        solve_nse(trange=trange, clearprvdata=debug, data_prfx=data_prefix,
                  vfileprfx=proutdir, pfileprfx=proutdir,
                  output_includes_bcs=True,
                  return_dictofvelstrs=True, return_dictofpstrs=True,
                  dictkeysstr=True,
                  start_ssstokes=True, **refmeshdict)

    return refvdict, refpdict


if __name__ == '__main__':
    problem = 'cylinderwake'
    scheme = 'CR'
    t0, tE, Nts = 0.0, 0.02, 16
    nswtchshortlist = [3, 2, 3]  # we recommend to start with `Nref`

    nswtchstr = 'Nswitches' + ''.join(str(e) for e in nswtchshortlist)
    dtstrdct = dict(prefix=ddir+problem+scheme+nswtchstr,
                    method=2, N=None, Nts=Nts, t0=t0, te=tE)
    check_the_sim(problem='cylinderwake', index=2, nswtchsl=nswtchshortlist,
                  scheme='CR', Re=120, Nts=Nts, paraout=False, t0=t0, tE=tE,
                  dtstrdct=dtstrdct, debug=debug)
