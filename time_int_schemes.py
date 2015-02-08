from dolfin import errornorm, TrialFunction, Function, assemble, div, dx, norm
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import dolfin_to_nparrays as dtn
import time

__all__ = ['halfexp_euler_smarminex',
           'halfexp_euler_nseind2',
           'comp_cont_error',
           'expand_vp_dolfunc',
           'get_conv_curfv_rearr',
           'pinthep']
try:
    import krypy.linsys
except ImportError:
    pass  # No krypy -- I hope we don't need it

try:
    def mass_fem_ip(q1, q2, M):
        """M^-1 inner product

        """
        try:
            # maybe M comes factorized
            miq2 = np.atleast_2d(M.solve(q2.flatten())).T
        except AttributeError:
            ls = krypy.linsys.LinearSystem(M, q2, self_adjoint=True)
            miq2 = (krypy.linsys.Cg(ls, tol=1e-12)).xk

        return np.dot(q1.T.conj(), miq2)

except NameError:
    pass  # no krypy -- I hope we don't need it


def halfexp_euler_smarminex(MSme, ASme, BSme, MP, FvbcSme, FpbcSme, B2BoolInv,
                            PrP, TsP, vp_init=None, qqpq_init=None,
                            saveallres=True):
    """ halfexplicit euler for the NSE in index 1 formulation

    """

    N = PrP.Pdof
    Nts, t0, tE, dt, Nv = init_time_stepping(PrP, TsP)
    tcur = t0
    # remove the p - freedom
    BSme, BTSme, MPc, FpbcSmeC, vp_init, Npc\
        = pinthep(BSme, BSme.T, MP, FpbcSme, vp_init, PrP.Pdof)

    # split the coeffs
    B1Sme = BSme[:, :-Npc]
    B2Sme = BSme[:, -Npc:]

    M1Sme = MSme[:, :-Npc]
    M2Sme = MSme[:, -Npc:]

    A1Sme = ASme[:, :-Npc]
    A2Sme = ASme[:, -Npc:]

    # The matrix to be solved in every time step
    #
    # 		1/dt*M11    M12  -B2'  0        q1
    # 		1/dt*M21    M22  -B1'  0        tq2
    # 		1/dt*B1     B2   0     0  *   p     = rhs
    # 		     B1     0    0     B2 	    q2
    #
    # cf. preprint
    # if A is there - we need to treat it implicitly
    #
    # 		1/dt*M11+A11    M12  -B2'  A12        q1
    # 		1/dt*M21+A21    M22  -B1'  A22        tq2
    # 		1/dt*B1         B2   0       0    *   p     = rhs
    # 		     B1         0    0       B2 	  q2
    #

    MFac = 1
    # Weights for the 'conti' eqns to balance the residuals
    WC = 0.5
    WCD = 0.5
    PFac = 1  # dt/WCD
    PFacI = 1  # WCD/dt

    # rescale q1
    q1facI = 1.

    IterA1 = MFac*sps.hstack([q1facI*(1.0/dt*M1Sme + A1Sme), M2Sme,
                              -PFacI*BSme.T, A2Sme])
    IterA2 = WCD*sps.hstack([q1facI*(1.0/dt*B1Sme), B2Sme,
                             sps.csr_matrix((Npc, 2*Npc))])
    IterA3 = WC*sps.hstack([q1facI*B1Sme, sps.csr_matrix((Npc, 2*Npc)), B2Sme])

    IterA = sps.vstack([IterA1, IterA2, IterA3])

    if TsP.linatol == 0:
        IterAfac = spsla.factorized(IterA)

    # Preconditioning ...
    #
    if TsP.SadPtPrec:
        MLump = np.atleast_2d(MSme.diagonal()).T
        MLump2 = MLump[-Npc:, ]
        MLumpI = 1. / MLump
        # MLumpI1 = MLumpI[:-(Np - 1), ]
        # MLumpI2 = MLumpI[-(Np - 1):, ]
        B2SmeTfac = spsla.splu(B2Sme.T)
        B2Smefac = spsla.splu(B2Sme)

        def PrecByB2(qqpq):
            qq = MLumpI*qqpq[:Nv, ]

            p = qqpq[Nv:-Npc, ]
            p = B2Smefac.solve(p.flatten())
            p = MLump2*np.atleast_2d(p).T
            # p = spsla.spsolve(B2Sme.T, p)
            p = B2SmeTfac.solve(p.flatten())
            p = np.atleast_2d(p).T

            q2 = qqpq[-Npc:, ]
            # q2 = spsla.spsolve(B2Sme, q2)
            q2 = B2Smefac.solve(q2.flatten())
            q2 = np.atleast_2d(q2).T

            return np.vstack([np.vstack([qq, -p]), q2])

        MGmr = spsla.LinearOperator(
            (Nv + 2*Npc,
             Nv + 2*Npc),
            matvec=PrecByB2,
            dtype=np.float32)
        TsP.Ml = MGmr

    Mcfac = spsla.splu(MSme)
    MPcfac = spsla.splu(MPc)

    def _MInvInd1(qqpq):
        qq = qqpq[:Nv, ]
        p = qqpq[Nv:-Npc, ]
        q2 = qqpq[-Npc:, ]
        miqq = np.atleast_2d(Mcfac.solve(qq.flatten())).T
        mip = np.atleast_2d(MPcfac.solve(p.flatten())).T
        miq2 = np.atleast_2d(MPcfac.solve(q2.flatten())).T
        return np.vstack([miqq, mip, miq2])

    # MInvInd1 = spsla.LinearOperator((Nv + 2*Npc, Nv + 2*Npc),
    #                                 matvec=_MInvInd1, dtype=np.float32)

    def smamin_prec_fem_ip(qqpq1, qqpq2, retparts=False):
        """ M ip for the preconditioned residuals

        """
        if retparts:
            return (np.dot(qqpq1[:Nv, ].T.conj(), MSme*qqpq2[:Nv, ]),
                    np.dot(qqpq1[Nv:-Npc, ].T.conj(), MPc*qqpq2[Nv:-Npc, ]),
                    np.dot(qqpq1[-Npc:, ].T.conj(), MPc*qqpq2[-Npc:, ]))
        else:
            return np.dot(qqpq1[:Nv, ].T.conj(), MSme*qqpq2[:Nv, ]) + \
                np.dot(qqpq1[Nv:-Npc, ].T.conj(), MPc*qqpq2[Nv:-Npc, ]) + \
                np.dot(qqpq1[-Npc:, ].T.conj(), MPc*qqpq2[-Npc:, ])

    def smamin_fem_ip(qqpq1, qqpq2, Mv, Mp, Nv, Npc, dt=1.):
        """ M^-1 ip for the extended system

        """
        return mass_fem_ip(qqpq1[:Nv, ], qqpq2[:Nv, ], Mv) + \
            dt*mass_fem_ip(qqpq1[Nv:-Npc, ], qqpq2[Nv:-Npc, ], Mp) + \
            mass_fem_ip(qqpq1[-Npc:, ], qqpq2[-Npc:, ], Mp)

    v, p = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=PrP.Pdof)
    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur

    dtstrdct = dict(prefix=TsP.svdatapath, method=1, N=PrP.N,
                    tolcor=TsP.TolCorB,
                    nu=PrP.nu, Nts=TsP.Nts, tol=TsP.linatol, te=TsP.tE)
    cdatstr = get_dtstr(t=0, **dtstrdct)
    try:
        np.load(cdatstr + '.npy')
        print 'loaded data from ', cdatstr, ' ...'
    except IOError:
        np.save(cdatstr, vp_init)
        print 'saving to ', cdatstr, ' ...'

    vp_old = np.copy(vp_init)
    q1_old = 1./q1facI*vp_init[~B2BoolInv, ]
    q2_old = vp_init[B2BoolInv, ]

    if qqpq_init is None and TsP.linatol > 0:
        # initial value for tq2
        ConV, CurFv = get_conv_curfv_rearr(v, PrP, tcur, B2BoolInv)
        tq2_old = spsla.spsolve(M2Sme[-Npc:, :], CurFv[-Npc:, ])
        # tq2_old = MLumpI2*CurFv[-(Np-1):,]
        tq2_old = np.atleast_2d(tq2_old).T
        # state vector of the smaminex system : [ q1^+, tq2^c, p^c, q2^+]
        qqpq_old = np.zeros((Nv + 2*Npc, 1))
        qqpq_old[:Nv - Npc, ] = q1_old
        qqpq_old[Nv - Npc:Nv, ] = tq2_old
        qqpq_old[Nv:Nv + Npc, ] = PFac*vp_old[Nv:, ]
        qqpq_old[Nv + Npc:, ] = q2_old
    else:
        qqpq_old = qqpq_init

    qqpq_oldold = qqpq_old

    ContiRes, VelEr, PEr, TolCorL = [], [], [], []
    MomRes, DContiRes = [], []

    # compute 1st time step by direct solve to initialize the krylov upd scheme
    inikryupd = TsP.inikryupd
    iniiterfac = TsP.iniiterfac  # the first krylov step needs more maxiter

    for etap in range(1, TsP.NOutPutPts + 1):
        for i in range(Nts / TsP.NOutPutPts):
            cdatstr = get_dtstr(t=tcur+dt, **dtstrdct)
            try:
                qqpq_next = np.load(cdatstr + '_qqpq' + '.npy')
                print 'loaded data from ', cdatstr, ' ...'
                qqpq_oldold = qqpq_old
                qqpq_old = qqpq_next
                TolCor = 'n.a.'
                if i == 2:
                    iniiterfac = 1  # fac only in the first Krylov Call

                # Reconstruct data of the iterative solution
                if TsP.linatol > 0 and (TsP.TolCorB or saveallres):
                    ConV, CurFv = get_conv_curfv_rearr(v, PrP, tcur, B2BoolInv)
                    gdot = np.zeros((Npc, 1))  # TODO: implement \dot g
                    Iterrhs = 1.0 / dt*np.vstack([MFac*M1Sme*q1_old,
                                                  WCD*B1Sme*q1_old]) +\
                        np.vstack([MFac*(FvbcSme + CurFv - ConV), WCD*gdot])
                    Iterrhs = np.vstack([Iterrhs, WC*FpbcSmeC])
                    if TsP.TolCorB:
                        NormRhsInd1 = np.sqrt(
                            smamin_fem_ip(Iterrhs, Iterrhs, Mcfac, MPcfac,
                                          Nv, Npc, dt=np.sqrt(dt)))[0][0]
                        TolCor = 1.0 / np.max([NormRhsInd1, 1])
                    else:
                        TolCor = 1.0

            except IOError:
                print 'computing data for ', cdatstr, ' ...'

                # set up right hand side
                ConV, CurFv = get_conv_curfv_rearr(v, PrP, tcur, B2BoolInv)
                gdot = np.zeros((Npc, 1))  # TODO: implement \dot g
                Iterrhs = 1.0 / dt*np.vstack([MFac*M1Sme*q1_old,
                                              WCD*B1Sme*q1_old]) +\
                    np.vstack([MFac*(FvbcSme + CurFv - ConV), WCD*gdot])
                Iterrhs = np.vstack([Iterrhs, WC*FpbcSmeC])

                if TsP.linatol == 0:
                    # q1_tq2_p_q2_new = spsla.spsolve(IterA, Iterrhs)
                    q1_tq2_p_q2_new = IterAfac(Iterrhs.flatten())
                    qqpq_old = np.atleast_2d(q1_tq2_p_q2_new).T
                    TolCor = 0
                else:
                    # Norm of rhs of index-1 formulation
                    # used to correct the relative residual
                    # such that the absolute residual stays constant
                    if TsP.TolCorB:
                        NormRhsInd1 = np.sqrt(
                            smamin_fem_ip(Iterrhs, Iterrhs, MSme, MPc,
                                          Nv, Npc, dt=np.sqrt(dt)))[0][0]
                        TolCor = 1.0 / np.max([NormRhsInd1, 1])
                    else:
                        TolCor = 1.0

                    if inikryupd and tcur == t0:
                        print '\n1st step direct solve to initialize krylov\n'
                        q1_tq2_p_q2_new = spsla.spsolve(IterA, Iterrhs)
                        qqpq_oldold = qqpq_old
                        qqpq_old = np.atleast_2d(q1_tq2_p_q2_new).T
                        TolCor = 0
                        inikryupd = False  # only once !!
                    else:
                        cls = krypy.linsys.\
                            LinearSystem(IterA, Iterrhs, Ml=MGmr,
                                         ip_B=smamin_prec_fem_ip)

                        tstart = time.time()
                        # extrapolating the initial value
                        qqqp_pv = (qqpq_old - qqpq_oldold)

                        q1_tq2_p_q2_new = krypy.linsys.\
                            RestartedGmres(cls, x0=qqpq_old+qqqp_pv,
                                           tol=TolCor*TsP.linatol,
                                           maxiter=iniiterfac*TsP.MaxIter,
                                           max_restarts=100)

                        qqpq_oldold = qqpq_old
                        qqpq_old = np.atleast_2d(q1_tq2_p_q2_new.xk)
                        tend = time.time()
                        print ('Needed {0} of max {4}*{1} iterations: ' +
                               'final relres = {2}\n TolCor was {3}').\
                            format(len(q1_tq2_p_q2_new.resnorms), TsP.MaxIter,
                                   q1_tq2_p_q2_new.resnorms[-1], TolCor,
                                   iniiterfac)
                        print 'Elapsed time {0}'.format(tend - tstart)
                        iniiterfac = 1  # fac only in the first Krylov Call

                np.save(cdatstr + '_qqpq', qqpq_old)

            q1_old = q1facI*qqpq_old[:Nv - Npc, ]
            q2_old = qqpq_old[-Npc:, ]

            # Extract the 'actual' velocity and pressure
            vc = np.zeros((Nv, 1))
            vc[~B2BoolInv, ] = q1_old
            vc[B2BoolInv, ] = q2_old
            # print np.linalg.norm(vc)

            pc = PFacI*qqpq_old[Nv:Nv + Npc, ]

            v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc,
                                     pdof=PrP.Pdof)

            cdatstr = get_dtstr(t=tcur+dt, **dtstrdct)
            np.save(cdatstr, np.vstack([vc, pc]))

            tcur += dt
            # the errors and residuals
            # ContiRes.append(comp_cont_error(v, FpbcSme, PrP.Q))
            TolCorL.append(TolCor)
            ContiRes.append(comp_cont_error(v, FpbcSme, PrP.Q))
            try:
                vCur, pCur = PrP.v, PrP.p
                vCur.t = tcur
                pCur.t = tcur - dt
                VelEr.append(errornorm(vCur, v))
                PEr.append(errornorm(pCur, p))
            except AttributeError:
                VelEr.append(0)
                PEr.append(0)

            if saveallres:
                res = IterA*qqpq_old - Iterrhs
                (mr, mcd, mc) = smamin_prec_fem_ip(res, res, retparts=True)
                MomRes.append(np.sqrt(mr)[0][0])
                DContiRes.append(np.sqrt(mcd)[0][0])
                ContiRes[-1] = np.sqrt(mc)[0][0]
                # print 'Res is dconti: ', np.sqrt(mcd)[0][0]
                # print 'Res is conti: ', np.sqrt(mc)[0][0]
                # print 'Res is moment: ', np.sqrt(mr+mc+mcd)[0][0]

            if i + etap == 1 and TsP.SaveIniVal:
                from scipy.io import savemat
                dname = 'IniValSmaMinN%s' % N
                savemat(dname, {'qqpq_old': qqpq_old})

        print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts)

        if TsP.ParaviewOutput:
            TsP.UpFiles.u_file << v, tcur
            TsP.UpFiles.p_file << p, tcur

    TsP.Residuals.VelEr.append(VelEr)
    TsP.Residuals.PEr.append(PEr)
    TsP.TolCor.append(TolCorL)
    TsP.Residuals.ContiRes.append(ContiRes)
    if saveallres:
        TsP.Residuals.MomRes.append(MomRes)
        TsP.Residuals.DContiRes.append(DContiRes)

    return


def halfexp_euler_nseind2(Mc, MP, Ac, BTc, Bc, fvbc, fpbc, PrP, TsP,
                          vp_init=None):
    """halfexplicit euler for the NSE in index 2 formulation
    """
    #
    #
    # Basic Eqn:
    #
    # 1/dt*M  -B.T    q+       1/dt*M*qc - K(qc) + fc
    #    B        * pc   =   g
    #
    #

    Nts, t0, tE, dt, Nv = init_time_stepping(PrP, TsP)

    tcur = t0

    MFac = dt
    CFac = 1  # /dt
    # PFac = -1  # -1 for symmetry (if CFac==1)
    PFacI = -1./dt

    dtstrdct = dict(prefix=TsP.svdatapath, method=2, N=PrP.N,
                    tolcor=TsP.TolCorB,
                    nu=PrP.nu, Nts=TsP.Nts, tol=TsP.linatol, te=TsP.tE)
    cdatstr = get_dtstr(t=0, **dtstrdct)
    try:
        np.load(cdatstr + '.npy')
        print 'loaded data from ', cdatstr, ' ...'
    except IOError:
        np.save(cdatstr, vp_init)
        print 'saving to ', cdatstr, ' ...'

    v, p = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur
    Bcc, BTcc, MPc, fpbcc, vp_init, Npc = pinthep(Bc, BTc, MP, fpbc,
                                                  vp_init, PrP.Pdof)

    IterAv = MFac*sps.hstack([1.0/dt*Mc + Ac, PFacI*(-1)*BTcc])
    IterAp = CFac*sps.hstack([Bcc, sps.csr_matrix((Npc, Npc))])
    IterA = sps.vstack([IterAv, IterAp])
    if TsP.linatol == 0:
        IterAfac = spsla.factorized(IterA)

    vp_old = vp_init
    vp_oldold = vp_old
    ContiRes, VelEr, PEr, TolCorL = [], [], [], []

    # Mvp = sps.csr_matrix(sps.block_diag((Mc, MPc)))
    # Mvp = sps.eye(Mc.shape[0] + MPc.shape[0])
    # Mvp = None

    # M matrix for the minres routine
    # M accounts for the FEM discretization

    Mcfac = spsla.splu(Mc)
    MPcfac = spsla.splu(MPc)

    def _MInv(vp):
        # v, p = vp[:Nv, ], vp[Nv:, ]
        # lsv = krypy.linsys.LinearSystem(Mc, v, self_adjoint=True)
        # lsp = krypy.linsys.LinearSystem(MPc, p, self_adjoint=True)
        # Mv = (krypy.linsys.Cg(lsv, tol=1e-14)).xk
        # Mp = (krypy.linsys.Cg(lsp, tol=1e-14)).xk
        v, p = vp[:Nv, ], vp[Nv:, ]
        Mv = np.atleast_2d(Mcfac(v.flatten())).T
        Mp = np.atleast_2d(MPcfac(p.flatten())).T
        return np.vstack([Mv, Mp])

    MInv = spsla.LinearOperator(
        (Nv + Npc,
         Nv + Npc),
        matvec=_MInv,
        dtype=np.float32)

    def ind2_ip(vp1, vp2):
        """

        for applying the fem inner product
        """
        v1, v2 = vp1[:Nv, ], vp2[:Nv, ]
        p1, p2 = vp1[Nv:, ], vp2[Nv:, ]
        return mass_fem_ip(v1, v2, Mcfac) + mass_fem_ip(p1, p2, MPcfac)

    inikryupd = TsP.inikryupd
    iniiterfac = TsP.iniiterfac  # the first krylov step needs more maxiter

    for etap in range(1, TsP.NOutPutPts + 1):
        for i in range(Nts / TsP.NOutPutPts):
            cdatstr = get_dtstr(t=tcur+dt, **dtstrdct)
            try:
                vp_next = np.load(cdatstr + '.npy')
                print 'loaded data from ', cdatstr, ' ...'
                vp_oldold = vp_old
                vp_old = vp_next
                if tcur == dt+dt:
                    iniiterfac = 1  # fac only in the first Krylov Call
            except IOError:
                print 'computing data for ', cdatstr, ' ...'
                ConV = dtn.get_convvec(v, PrP.V)
                CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

                Iterrhs = np.vstack([MFac*1.0/dt*Mc*vp_old[:Nv, ],
                                     np.zeros((Npc, 1))]) +\
                    np.vstack([MFac*(fvbc + CurFv - ConV[PrP.invinds, ]),
                               CFac*fpbcc])

                if TsP.linatol == 0:
                    # ,vp_old,tol=TsP.linatol)
                    vp_new = IterAfac(Iterrhs.flatten())
                    # vp_new = spsla.spsolve(IterA, Iterrhs)
                    vp_old = np.atleast_2d(vp_new).T
                    TolCor = 0

                else:
                    if inikryupd and tcur == t0:
                        print '\n1st step direct solve to initialize krylov\n'
                        vp_new = spsla.spsolve(IterA, Iterrhs)
                        vp_old = np.atleast_2d(vp_new).T
                        TolCor = 0
                        inikryupd = False  # only once !!
                    else:
                        if TsP.TolCorB:
                            NormRhsInd2 = \
                                np.sqrt(ind2_ip(Iterrhs, Iterrhs))[0][0]
                            TolCor = 1.0 / np.max([NormRhsInd2, 1])
                        else:
                            TolCor = 1.0

                        curls = krypy.linsys.LinearSystem(IterA, Iterrhs,
                                                          M=MInv)

                        tstart = time.time()

                        # extrapolating the initial value
                        upv = (vp_old - vp_oldold)

                        ret = krypy.linsys.\
                            RestartedGmres(curls, x0=vp_old + upv,
                                           tol=TolCor*TsP.linatol,
                                           maxiter=iniiterfac*TsP.MaxIter,
                                           max_restarts=100)

                        # ret = krypy.linsys.\
                        #     Minres(curls, maxiter=20*TsP.MaxIter,
                        #            x0=vp_old + upv, tol=TolCor*TsP.linatol)
                        tend = time.time()
                        vp_oldold = vp_old
                        vp_old = ret.xk

                        print ('Needed {0} of max {4}*{1} iterations: ' +
                               'final relres = {2}\n TolCor was {3}').\
                            format(len(ret.resnorms), TsP.MaxIter,
                                   ret.resnorms[-1], TolCor, iniiterfac)
                        print 'Elapsed time {0}'.format(tend - tstart)
                        iniiterfac = 1  # fac only in the first Krylov Call

                np.save(cdatstr, np.vstack([vp_old[:Nv],
                                            PFacI*vp_old[Nv:]]))

            vc = vp_old[:Nv, ]
            print 'Norm of current v: ', np.linalg.norm(vc)
            pc = PFacI*vp_old[Nv:, ]

            v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc)

            tcur += dt

            # the errors
            vCur, pCur = PrP.v, PrP.p
            try:
                vCur.t = tcur
                pCur.t = tcur - dt

                ContiRes.append(comp_cont_error(v, fpbc, PrP.Q))
                VelEr.append(errornorm(vCur, v))
                PEr.append(errornorm(pCur, p))
                TolCorL.append(TolCor)
            except AttributeError:
                ContiRes.append(0)
                VelEr.append(0)
                PEr.append(0)
                TolCorL.append(0)

        print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts)

        if TsP.ParaviewOutput:
            TsP.UpFiles.u_file << v, tcur
            TsP.UpFiles.p_file << p, tcur

    TsP.Residuals.ContiRes.append(ContiRes)
    TsP.Residuals.VelEr.append(VelEr)
    TsP.Residuals.PEr.append(PEr)
    TsP.TolCor.append(TolCorL)

    return


def comp_cont_error(v, fpbc, Q):
    """Compute the L2 norm of the residual of the continuity equation
            Bv = g
    """

    q = TrialFunction(Q)
    divv = assemble(q*div(v)*dx)

    conRhs = Function(Q)
    conRhs.vector().set_local(fpbc)

    ContEr = norm(conRhs.vector() - divv)

    return ContEr


def expand_vp_dolfunc(PrP, vp=None, vc=None, pc=None, pdof=None):
    """expand v and p to the dolfin function representation

    pdof = pressure dof that was set zero
    """

    v = Function(PrP.V)
    p = Function(PrP.Q)

    if vp is not None:
        if vp.ndim == 1:
            vc = vp[:len(PrP.invinds)].reshape(len(PrP.invinds), 1)
            pc = vp[len(PrP.invinds):].reshape(PrP.Q.dim() - 1, 1)
        else:
            vc = vp[:len(PrP.invinds), :]
            pc = vp[len(PrP.invinds):, :]

    ve = np.zeros((PrP.V.dim(), 1))

    # fill in the boundary values
    for bc in PrP.velbcs:
        bcdict = bc.get_boundary_values()
        ve[bcdict.keys(), 0] = bcdict.values()

    ve[PrP.invinds] = vc

    if pdof is None:
        pe = pc
    elif pdof == 0:
        pe = np.vstack([[0], pc])
    elif pdof == -1:
        pe = np.vstack([pc, [0]])
    else:
        pe = np.vstack([pc[:pdof], np.vstack([[0], pc[pdof:]])])

    v.vector().set_local(ve)
    p.vector().set_local(pe)

    v.rename("v", "field")
    p.rename("p", "field")

    return v, p


def get_conv_curfv_rearr(v, PrP, tcur, B2BoolInv):

    ConV = dtn.get_convvec(v, PrP.V)
    ConV = ConV[PrP.invinds, ]

    ConV = np.vstack([ConV[~B2BoolInv], ConV[B2BoolInv]])

    CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)
    if len(CurFv) != len(PrP.invinds):
        raise Warning('Need fv at innernodes here')
    CurFv = np.vstack([CurFv[~B2BoolInv], CurFv[B2BoolInv]])

    return ConV, CurFv


def init_time_stepping(PrP, TsP):
    """what every method starts with """

    Nts, t0, tE = TsP.Nts, TsP.t0, TsP.tE
    dt = (tE - t0) / Nts
    Nv = len(PrP.invinds)

    if Nts % TsP.NOutPutPts != 0:
        TsP.NOutPutPts = 1

    return Nts, t0, tE, dt, Nv


def pinthep(B, BT, M, fp, vp_init, pdof):
    """remove dofs of div and grad to pin the pressure

    """
    (NP, NV) = B.shape
    if pdof is None:
        return B, BT, M, fp, vp_init, NP
    elif pdof == 0:
        vpi = np.vstack([vp_init[:NV, :], vp_init[NV+1:, :]])
        return (B[1:, :], BT[:, 1:], M[1:, :][:, 1:], fp[1:, :],
                vpi, NP - 1)
    elif pdof == -1:
        return (B[:-1, :], BT[:, :-1], M[:-1, :][:, :-1],
                fp[:-1, :], vp_init[:-1, :], NP - 1)
    else:
        raise NotImplementedError()


def get_dtstr(t=None, prefix='', method=None, N=None,
              nu=None, Nts=None, tol=None, te=None, tolcor=None, **kwargs):
    return prefix + '_m{0}_N{1}_nu{2}_Nts{3}_tol{4}_tolcor{7}_te{6}_t{5}'.\
        format(method, N, nu, Nts, tol, t, te, tolcor)
