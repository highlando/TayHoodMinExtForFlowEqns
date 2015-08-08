import numpy as np

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

from time_int_schemes import get_dtstr


def rothe_ind2(problem='cylinderwake', nu=None, Re=None,
               Nts=256, t0=0.0, tE=0.2, Nlist=[2],
               viniv=None, piniv=None, Nini=None,
               scheme=None, dtstrdct={}):

    trange = np.linspace(t0, tE, Nts+1)
    # set up the list of the mesh parameters at every time step
    swl = Nts/len(Nlist)
    Nll = [Nlist[0]]*Nts
    for N in Nlist[1:]:
        Nll[N*swl+1:] = N
    Nll[0] = Nini

    t = trange[0]
    dtstrdct.update(dict(t=0, N=Nll[0]))
    cdatstr = get_dtstr(**dtstrdct)
    dou.save_npa(viniv, cdatstr + '__vel')
    curvdict = {0: cdatstr + '__vel'}
    dou.save_npa(piniv, cdatstr + '__p')
    curpdict = {0: cdatstr + '__p'}

    vprev = viniv
    curmeshdict = get_curmeshdict(problem=problem, N=Nll[0], nu=nu, Re=Re,
                                  scheme=scheme)
    curmeshdict.update(coefalu=None)
    for tk, t in enumerate(trange[1:]):
        cts = t - trange[tk]
        Nvc = curmeshdict['A'].shape[0]
        vpcur, coefalu = \
            roth_upd_ind2(vvec=vprev, cts=cts,
                          Vc=curmeshdict['V'], diribcsc=curmeshdict['diribcs'],
                          nmd=curmeshdict, returnalu=True)
        dtstrdct.update(dict(t=t, N=N))
        cdatstr = get_dtstr(**dtstrdct)
        vcur = dts.append_bcs_vec(vpcur[:Nvc], **curmeshdict)
        dou.save_npa(vcur, cdatstr+'__vel')
        curvdict.update({t: cdatstr})
        dou.save_npa(vpcur[Nvc:, :], cdatstr+'__p')
        curpdict.update({t: cdatstr})
        curmeshdict.update(dict(coefalu=coefalu))

    return curvdict, curpdict


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


def _vctovn(vvec=None, vfunc=None, Vc=None, Vn=None, diribcs=None):
    return vvec


def get_curmeshdict(problem=None, N=None, Re=None, nu=None, scheme=None):

    femp, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problem, N=N,
                                                Re=Re, nu=nu, scheme=scheme,
                                                mergerhs=True)
    M, A, J = stokesmatsc['M'], stokesmatsc['A'], stokesmatsc['J']
    V, Q = femp['V'], femp['Q']
    invinds, diribcs = femp['invinds'],  femp['diribcs']
    fv, fp = rhsd['fv'], rhsd['fp']

    return dict(M=M, A=A, J=J, V=V, Q=Q, invinds=invinds, diribcs=diribcs,
                fv=fv, fp=fp, N=N, Re=femp['Re'])
