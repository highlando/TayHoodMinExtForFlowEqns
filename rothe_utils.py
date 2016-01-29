import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

from time_int_schemes import get_dtstr
import smamin_thcr_mesh as smt

import logging
logger = logging.getLogger("rothemain.rothe_utils")

__all__ = ['roth_upd_smmx',
           'rothe_time_int',
           'roth_upd_ind2',
           'get_curmeshdict',
           'plottimeerrs']


def roth_upd_smmx(vvec=None, cts=None, nu=None, Vc=None, diribcsc=None,
                  nmd=dict(V=None, Q=None, M=None, invinds=None, npc=None,
                           MSme=None, ASme=None, JSme=None, fvSme=None,
                           fp=None, diribcs=None, coefalu=None,
                           smmxqq2vel=None, vel2smmxqq=None),
                  returnalu=False, **kwargs):
    """ advancing `v, p` for one time using Rothe's method

    Parameters
    ---
    vvec : (n, 1) array
        the current velocity solution vector incl. bcs in the actual coors
    nmd : dict
        containing the data (mesh, matrices, rhs) from the next time step,
        with the `*Sme` matrices resorted according to the minimal extension
    vvec : (n,1)-array
        current solution
    Vc : dolfin.mesh
        current mesh

    Notes
    -----
    Time dependent Dirichlet conditions are not supported by now
    """

    Npc = nmd['npc']
    # split the coeffs
    J1Sme = nmd['JSme'][:, :-Npc]
    J2Sme = nmd['JSme'][:, -Npc:]

    M1Sme = nmd['MSme'][:, :-Npc]
    M2Sme = nmd['MSme'][:, -Npc:]

    A1Sme = nmd['ASme'][:, :-Npc]
    A2Sme = nmd['ASme'][:, -Npc:]

    if not nmd['V'] == Vc:
        vvec = _vctovn(vvec=vvec, Vc=Vc, Vn=nmd['V'])
        logger.debug('len(vvec)={0}, dim(Vn)={1}, dim(Vc)={2}'.
                     format(vvec.size, nmd['V'].dim(), Vc.dim()))
    q1c, q2c = nmd['vel2smmxqq'](vvec[nmd['invinds']])

    convvec = dts.get_convvec(u0_vec=vvec, V=nmd['V'],
                              diribcs=nmd['diribcs'],
                              invinds=nmd['invinds'])

    logger.debug('in `roth_upd_smmx`: cts={0}'.format(cts))

    coefmatmom = sps.hstack([1/cts*M1Sme+A1Sme, M2Sme, -nmd['JSme'].T, A2Sme])
    coefmdivdrv = sps.hstack([1/cts*J1Sme, J2Sme,
                              sps.csr_matrix((Npc, 2*Npc))])
    coefmdiv = sps.hstack([J1Sme, sps.csr_matrix((Npc, 2*Npc)), J2Sme])
    coefmat = sps.vstack([coefmatmom, coefmdivdrv, coefmdiv])

    rhsmom = 1/cts*M1Sme*q1c - nmd['vel2smmxqq'](convvec, getitstacked=True) +\
        nmd['fvSme']
    rhsdivdrv = 1/cts*J1Sme*q1c
    rhsdiv = nmd['fp']
    rhs = np.vstack([rhsmom, rhsdivdrv, rhsdiv])

    qqpqnext = spsla.spsolve(coefmat, rhs)

    Nvc = A1Sme.shape[0]
    q1, q2 = qqpqnext[:Nvc-Npc], qqpqnext[Nvc+Npc:Nvc+2*Npc]
    p_new = qqpqnext[Nvc:Nvc+Npc]
    v_new = nmd['smmxqq2vel'](q1=q1, q2=q2)
    vp_new = np.vstack([v_new, p_new.reshape((p_new.size, 1))])

    if returnalu:
        return vp_new, None
    else:
        return vp_new


def rothe_time_int(problem='cylinderwake', nu=None, Re=None,
                   Nts=256, t0=0.0, tE=0.2, Nll=[2],
                   viniv=None, piniv=None, Nini=None,
                   scheme=None, dtstrdct={}, method=2):

    trange = np.linspace(t0, tE, Nts+1)

    t = trange[0]
    dtstrdct.update(dict(t=0, N=Nll[0]))
    cdatstr = get_dtstr(**dtstrdct)
    dou.save_npa(viniv, cdatstr + '__vel')
    curvdict = {t: cdatstr + '__vel'}
    dou.save_npa(piniv, cdatstr + '__p')
    logger.info('v/p saved to ' + cdatstr + '__v/__p')
    curpdict = {t: cdatstr + '__p'}
    smaminex = True if method == 1 else False

    vcurvec = viniv
    logger.debug(' t={0}, |v|={1}'.format(t, npla.norm(vcurvec)))
    curmeshdict = get_curmeshdict(problem=problem, N=Nll[0], nu=nu, Re=Re,
                                  scheme=scheme, smaminex=smaminex)
    curmeshdict.update(coefalu=None)
    Vc = curmeshdict['V']
    for tk, t in enumerate(trange[1:]):
        cts = t - trange[tk]
        if not Nll[tk+1] == Nll[tk]:
            curmeshdict = get_curmeshdict(problem=problem, N=Nll[tk+1], nu=nu,
                                          Re=Re, scheme=scheme,
                                          smaminex=smaminex)
            curmeshdict.update(coefalu=None)
            logger.info('changed the mesh from N={0} to N={1} at t={2}'.
                        format(Nll[tk], Nll[tk+1], t))
            # change in the mesh
        Nvc = curmeshdict['A'].shape[0]
        logger.debug("t={0}, dim V={1}".format(t, curmeshdict['V'].dim()))
        if smaminex:
            vpcur, coefalu = \
                roth_upd_smmx(vvec=vcurvec, cts=cts,
                              Vc=Vc, diribcsc=curmeshdict['diribcs'],
                              nmd=curmeshdict, returnalu=True)
        else:  # index 2
            vpcur, coefalu = \
                roth_upd_ind2(vvec=vcurvec, cts=cts,
                              Vc=Vc, diribcsc=curmeshdict['diribcs'],
                              nmd=curmeshdict, returnalu=True)
        dtstrdct.update(dict(t=t, N=Nll[tk+1]))
        cdatstr = get_dtstr(**dtstrdct)
        # add the boundary values to the velocity
        vcurvec = dts.append_bcs_vec(vpcur[:Nvc], **curmeshdict)
        logger.debug(' t={0}, |v|={1}'.format(t, npla.norm(vcurvec)))
        dou.save_npa(vcurvec, cdatstr+'__vel')
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
    logger.debug("length of vvec={0}".format(vvec.size))
    if not nmd['V'] == Vc:
        vvec = _vctovn(vvec=vvec, Vc=Vc, Vn=nmd['V'])
        logger.debug('len(vvec)={0}, dim(Vn)={1}, dim(Vc)={2}'.
                     format(vvec.size, nmd['V'].dim(), Vc.dim()))
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


class ExtFunZero(dolfin.Expression):
    """a dolfin.expression that equals a function on its mesh and =0 elsewhere


    """
    def __init__(self, vfun=None):
        self.vfun = vfun

    def eval(self, value, x):
        try:
            self.vfun.eval(value, x)
        except RuntimeError:
            value[0] = 0.0
            value[1] = 0.0
            logger.debug("extfunzero: got x={0}, gave value={1}".
                         format(x, value))

    def value_shape(self):
        return (2,)


def _vctovn(vvec=None, vfun=None, Vc=None, Vn=None, diribcs=None):

    if vfun is None:
        vcfun = dolfin.Function(Vc)
        vcfun.vector().set_local(vvec)
    else:
        vcfun = vfun
    extvcfun = ExtFunZero(vfun=vcfun)
    vnfun = dolfin.interpolate(extvcfun, Vn)
    vnvec = vnfun.vector().array().reshape((Vn.dim(), 1))

    return vnvec


def get_curmeshdict(problem=None, N=None, Re=None, nu=None, scheme=None,
                    onlymesh=False, smaminex=False):
    """

    Parameters
    ---
    smaminex : boolean, optional
        whether compute return the rearranged matrices needed for the
        index-1 formulation with minimal extension [Altmann, Heiland 2015],
        defaults to `False`
    """

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
        nu = femp['nu'] if nu is None else nu
        fv, fp = rhsd['fv'], rhsd['fp']
        cmd = dict(M=M, A=A, J=J, V=V, Q=Q, invinds=invinds, diribcs=diribcs,
                   fv=fv, fp=fp, N=N, Re=femp['Re'])
        logger.debug('in `get_curmeshdict`: ' +
                     'problem={0}, scheme={1}, Re={2}, nu={3}, ppin={4}'.
                     format(problem, scheme, Re, nu, femp['ppin']))
        if smaminex:
            cricelldict = {0: 758, 1: 1498, 2: 2386, 3: 4843}
            if problem == 'cylinderwake' and scheme == 'CR':
                try:
                    cricell = cricelldict[N]
                except KeyError():
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            MSmeCL, ASmeCL, BSme, B2Inds, B2BoolInv, B2BI = smt.\
                get_smamin_rearrangement(N, None, M=M, A=A, B=J,
                                         V=V, Q=Q, nu=nu, mesh=femp['mesh'],
                                         crinicell=cricell, addnedgeat=cricell,
                                         Pdof=femp['ppin'],
                                         scheme=scheme, invinds=invinds,
                                         fullB=stokesmatsc['Jfull'])

            FvbcSme = np.vstack([fv[~B2BoolInv, ], fv[B2BoolInv, ]])

            def sortitback(q1=None, q2=None):
                vc = np.zeros((fv.size, 1))
                vc[~B2BoolInv, ] = q1.reshape((q1.size, 1))
                vc[B2BoolInv, ] = q2.reshape((q2.size, 1))
                return vc

            def sortitthere(vc, getitstacked=False):
                vc = np.zeros((fv.size, 1))
                q1 = vc[~B2BoolInv, ]
                q2 = vc[B2BoolInv, ]
                if getitstacked:
                    return np.vstack([q1, q2])
                else:
                    return q1, q2

            cmd.update(dict(ASme=ASmeCL, JSme=BSme,
                            MSme=MSmeCL, fvSme=FvbcSme,
                            smmxqq2vel=sortitback, npc=BSme.shape[0],
                            vel2smmxqq=sortitthere))

        return cmd


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
