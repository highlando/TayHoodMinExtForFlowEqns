import numpy as np
import matplotlib.pyplot as plt

import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

from time_int_schemes import get_dtstr

import logging
logger = logging.getLogger("rothemain.rothe_utils")


def rothe_ind2(problem='cylinderwake', nu=None, Re=None,
               Nts=256, t0=0.0, tE=0.2, Nll=[2],
               viniv=None, piniv=None, Nini=None,
               scheme=None, dtstrdct={}):

    trange = np.linspace(t0, tE, Nts+1)

    t = trange[0]
    dtstrdct.update(dict(t=0, N=Nll[0]))
    cdatstr = get_dtstr(**dtstrdct)
    dou.save_npa(viniv, cdatstr + '__vel')
    curvdict = {t: cdatstr + '__vel'}
    dou.save_npa(piniv, cdatstr + '__p')
    logger.info('v/p saved to ' + cdatstr + '__v/__p')
    curpdict = {t: cdatstr + '__p'}

    vcurvec = viniv
    curmeshdict = get_curmeshdict(problem=problem, N=Nll[0], nu=nu, Re=Re,
                                  scheme=scheme)
    curmeshdict.update(coefalu=None)
    Vc = curmeshdict['V']
    for tk, t in enumerate(trange[1:]):
        cts = t - trange[tk]
        if not Nll[tk+1] == Nll[tk]:
            curmeshdict = get_curmeshdict(problem=problem, N=Nll[tk+1], nu=nu,
                                          Re=Re, scheme=scheme)
            logger.info('changed the mesh from N={0} to N={1} at t={2}'.
                        format(Nll[tk], Nll[tk+1], t))
            # change in the mesh
        Nvc = curmeshdict['A'].shape[0]
        logger.debug("t={0}, dim V={1}".format(t, curmeshdict['V'].dim()))
        vpcur, coefalu = \
            roth_upd_ind2(vvec=vcurvec, cts=cts,
                          Vc=Vc, diribcsc=curmeshdict['diribcs'],
                          nmd=curmeshdict, returnalu=True)
        dtstrdct.update(dict(t=t, N=Nll[tk+1]))
        cdatstr = get_dtstr(**dtstrdct)
        # add the boundary values to the velocity
        vcur = dts.append_bcs_vec(vpcur[:Nvc], **curmeshdict)
        dou.save_npa(vcur, cdatstr+'__vel')
        curvdict.update({t: cdatstr+'__vel'})
        dou.save_npa(vpcur[Nvc:, :], cdatstr+'__p')
        curpdict.update({t: cdatstr+'__p'})
        curmeshdict.update(dict(coefalu=coefalu))
        Vc = curmeshdict['V']

    return curvdict, curpdict


def roth_upd_ind2(vvec=None, cts=None, nu=None, Vc=None, diribcsc=None,
                  nmd=dict(V=None, Q=None,
                           M=None, A=None, J=None, fv=None, fp=None,
                           invinds=None, diribcs=None, coefalu=None),
                  returnalu=False, **kwargs):
    """ advancing `v, p` for one time using Rothe's method

    Parameters
    ---
    nmd : dict
        containing the data (mesh, matrices, rhs) from the next time step
    vvec : (n,1)-array
        current solution
    Vc : dolfin.mesh
        current mesh

    Notes
    -----
    Time dependent Dirichlet conditions are not supported by now
    """
    logger.debug("dimension of vcur={0}".format(vvec.size))
    if not nmd['V'] == Vc:
        vvec = _vctovn(vvec=vvec, Vc=Vc, Vn=nmd['V'])

    mvvec = nmd['M']*vvec[nmd['invinds'], :]
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


def _vctovn(vvec=None, vfun=None, Vc=None, Vn=None, diribcs=None):

    if vfun is None:
        vcfun = dolfin.Function(Vc)
        vcfun.vector().set_local(vvec)
    else:
        vcfun = vfun
    dolfin.parameters["allow_extrapolation"] = True
    vnfun = dolfin.interpolate(vcfun, Vn)
    vnvec = vnfun.vector().array().reshape((Vn.dim(), 1))

    return vnvec


def get_curmeshdict(problem=None, N=None, Re=None, nu=None, scheme=None,
                    onlymesh=False):

    if onlymesh:
        femp = dnsps.get_sysmats(problem=problem, N=N, scheme=scheme,
                                 onlymesh=True)
        return femp
    else:
        femp, stokesmatsc, rhsd = dnsps.\
            get_sysmats(problem=problem, N=N, Re=Re, nu=nu,
                        scheme=scheme, mergerhs=True)
        M, A, J = stokesmatsc['M'], stokesmatsc['A'], stokesmatsc['J']
        V, Q = femp['V'], femp['Q']
        invinds, diribcs = femp['invinds'],  femp['diribcs']
        fv, fp = rhsd['fv'], rhsd['fp']

        return dict(M=M, A=A, J=J, V=V, Q=Q, invinds=invinds, diribcs=diribcs,
                    fv=fv, fp=fp, N=N, Re=femp['Re'])


def plottimeerrs(trange=None, verrl=None, perrl=None, fignums=[131, 132],
                 showplot=False):
    if verrl is not None:
        plt.figure(fignums[0])
        for verr in verrl:
            plt.plot(trange, verr)
        plt.title('v error')
    if perrl is not None:
        plt.figure(fignums[1])
        for perr in perrl:
            plt.plot(trange, perr)
        plt.title('p error')
    if showplot:
        plt.show(block=False)
