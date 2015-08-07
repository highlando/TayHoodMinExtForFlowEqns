import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

from time_int_schemes import expand_vp_dolfunc, get_dtstr

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

'''

Nref = 3
Nplt = 2

nswtchl = [3, 2]

proutdir = 'results/'
ddir = 'data/'

nswtchstr = 'Nswitches' + ''.join(str(e) for e in nswtchl)


def gettheref(problem='drivencavity', N=None, nu=None, Re=None, Nts=None,
              paraout=False, t0=0.0, tE=0.2, scheme=None, dtstrdct={}):

    femp, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problem, N=N,
                                                Re=Re, nu=nu, scheme=scheme,
                                                mergerhs=True)
    trange = np.linspace(t0, tE, Nts+1)
    M, A, J, = stokesmatsc['M'], stokesmatsc['A'], stokesmatsc['J']
    V, Q = femp['V'], femp['Q']
    invindsref, diribcsref = femp['invinds'],  femp['diribcs']
    fv, fp = rhsd['fv'], rhsd['fp']

    refvdict, refpdict = snu.\
        solve_nse(A=A, M=M, J=J, JT=None, fv=fv, fp=fp, trange=trange,
                  V=V, Q=Q, invinds=invindsref, diribcs=diribcsref,
                  N=N, nu=nu,
                  clearprvdata=False, data_prfx=ddir,
                  vfileprfx=proutdir, pfileprfx=proutdir,
                  return_dictofvelstrs=True, return_dictofpstrs=True,
                  start_ssstokes=True)

    return refvdict, refpdict


def testit(problem='drivencavity', N=None, nu=None, Re=None, Nts=256,
           paraout=False, t0=0.0, tE=0.2, scheme=None, dtstrdct={}):

    femp, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problem, N=N,
                                                Re=Re, nu=nu, scheme=scheme,
                                                mergerhs=True)
    trange = np.linspace(t0, tE, Nts+1)

    # compute the reference solution

    viniv = dou.load_npa(refvdict[0])
    piniv = dou.load_npa(refpdict[0])

    vini, pini = dts.expand_vp_dolfunc(V=Vref, Q=Qref, vc=viniv,
                                       pc=piniv, invinds=invindsref,
                                       diribcs=diribcsref, ppin=-1)

    t = trange[0]
    # if paraout:
    #     vfile << vini, t
    #     pfile << pini, t
    dtstrdct.update(dict(t=0, N=Nref))
    cdatstr = get_dtstr(**dtstrdct)
    dou.save_npa(np.vstack([viniv, piniv]), cdatstr)
    cursoldict = {0: cdatstr}

    vprev = viniv
    coefalu = None
    curmeshdict = dict(V=Vref, Q=Qref, invinds=invindsref, diribcs=diribcsref,
                       fv=fv, fp=fp, A=A, M=M, J=J, coefalu=None)
    for tk, t in enumerate(trange[1:]):
        cts = t - trange[tk]
        vpcur, coefalu = \
            roth_upd_ind2(vvec=vprev, cts=cts,
                          Vc=curmeshdict['V'], diribcsc=curmeshdict['diribcs'],
                          nmd=curmeshdict, returnalu=True)
        dtstrdct.update(dict(t=t, N=N))
        dou.save_npa(vpcur, get_dtstr(**dtstrdct))
        curmeshdict.update(dict(coefalu=coefalu))
        cursoldict.update({t: cdatstr})

    return cursoldict



def roth_upd_ind2(vvec=None, cts=None, nu=None, Vc=None, diribcsc=None,
                  nmd=dict(V=None, Q=None,
                           M=None, A=None, J=None, fv=None, fp=None,
                           invinds=None, diribcs=None, coefalu=None),
                  returnalu=False, **kwargs):
    """ advancing `v, p` for one time using Rothe's method

    Notes
    -----
    Time dependent Dirichlet conditions are not supported by now
    """

    if not nmd['V'] == Vc:
        vvec = _vctovn(vvec=vvec, Vc=Vc, diribcs=diribcsc, nmd=nmd)

    mvvec = nmd['M']*vvec
    convvec = dts.get_convvec(u0_vec=vvec, V=nmd['V'],
                              diribcs=nmd['diribcs'],
                              invinds=nmd['invinds'])
    if nmd['coefalu'] is None:
        mta = nmd['M'] + cts*nmd['A']
        mtJT = -cts*nmd['J'].T
    else:
        mta = None
        mtJT = None

    rhsv = mvvec + cts*(nmd['fv'] - convvec)

    lsdpdict = dict(amat=mta, jmat=nmd['J'], jmatT=mtJT, rhsv=rhsv,
                    rhsp=nmd['fp'], sadlu=nmd['coefalu'],
                    return_alu=returnalu)
    if returnalu:
        vp_new, coefalu = lau.solve_sadpnt_smw(**lsdpdict)
        return vp_new, coefalu
    else:
        vp_new = lau.solve_sadpnt_smw(**lsdpdict)
        return vp_new


def _vctovn(vvec=None, Vc=None, Vn=None, diribcs=None):
    return vvec

if __name__ == '__main__':
    problem = 'cylinderwake'
    scheme = 'CR'
    t0, tE, Nts = 0.0, 0.2, 128
    dtstrdct = dict(prefix=ddir+problem+scheme+'_Rothe_velpres_'+nswtchstr,
                    method=2, N=None, nu=nu, Nts=Nts, t0=t0, te=tE)

    testit(problem=problem, N=3, Re=120, Nts=128, tE=.2,
           scheme='CR', dtstrdct=dtstrdct)
