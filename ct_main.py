import dolfin
# import plot_utils as plu

import numpy as np
# import scipy.sparse as sps
import matplotlib.pyplot as plt
import os
import glob

import dolfin_to_nparrays as dtn
import time_int_schemes as tis
import smartminex_tayhoomesh as smt

from prob_defs import ProbParams, FempToProbParams

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu


class TimestepParams(object):

    def __init__(self, method, N, scheme=None):
        self.t0 = 0
        self.tE = 1.0
        self.Omega = 8
        self.Ntslist = [128]
        self.NOutPutPts = 16
        self.method = method
        self.SadPtPrec = True
        self.UpFiles = UpFiles(method, scheme=scheme)
        self.Residuals = NseResiduals()
        self.linatol = 0  # 1e-4  # 0 for direct sparse solver
        self.inikryupd = True  # initialization of krylov upd scheme
        self.iniiterfac = 4  # often the first iteration needs more maxiters
        self.TolCor = []
        self.MaxIter = 85
        self.Ml = None  # preconditioners
        self.Mr = None
        self.ParaviewOutput = False
        self.SaveIniVal = False
        self.SaveTStps = False
        self.TolCorB = False
        self.svdatatdsc = True
        self.svdatapath = 'data/'


def solve_euler_timedep(method=1, Omega=8, tE=None, Prec=None,
                        N=40, NtsList=None, LinaTol=None, MaxIter=None,
                        UsePreTStps=None, SaveTStps=None, SaveIniVal=None,
                        scheme='TH', nu=0, Re=None, inikryupd=None,
                        tolcor=False, prob=None):
    """system to solve

             du\dt + (u*D)u + grad p = fv
                      div u          = fp

    """

    methdict = {
        1: 'HalfExpEulSmaMin',
        2: 'HalfExpEulInd2'}

    # instantiate object containing mesh, V, Q, rhs, velbcs, invinds
    # set nu=0 for Euler flow
    if prob == 'cyl':
        femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
            = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                                scheme=scheme)

        Mc, Ac = stokesmatsc['M'], stokesmatsc['A']
        MPa = stokesmatsc['MP']
        BTc, Bc = stokesmatsc['JT'], stokesmatsc['J']
        Ba = stokesmatsc['Jfull']

        bcinds, bcvals = femp['bcinds'], femp['bcvals']

        fvbc, fpbc = rhsd_stbc['fv'], rhsd_stbc['fp']
        inivdict = dict(A=Ac, J=Bc, JT=BTc, M=Mc,
                        ppin=None, V=femp['V'], Q=femp['Q'],
                        fv=fvbc, fp=fpbc, vel_pcrd_stps=0, vel_nwtn_stps=0,
                        return_vp=True, diribcs=femp['diribcs'],
                        invinds=femp['invinds'])
        dimredsys = Bc.shape[1] + Bc.shape[0]
        vp_init = snu.solve_steadystate_nse(**inivdict)[0]

        PrP = FempToProbParams(N, omega=Omega, femp=femp, pdof=None)
        PrP.Pdof = None  # No p pinning for outflow flow

        print 'Nv, Np -- w/o boundary nodes', BTc.shape
    else:
        if Re is not None:
            nu = 1./Re
        PrP = ProbParams(N, omega=Omega, nu=nu, scheme=scheme)
        # get system matrices as np.arrays
        Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q, nu=nu)
        fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)
        print 'Nv, Np -- w/ boundary nodes', BTa.shape

        # condense the system by resolving the boundary values
        (Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals,
         invinds) = dtn.condense_sysmatsbybcs(Ma, Aa, BTa, Ba,
                                              fv, fp, PrP.velbcs)
        print 'Nv, Np -- w/o boundary nodes', BTc.shape
        PrP.Pdof = 0  # Thats how the smamin is constructed

        dimredsys = Bc.shape[0] + Bc.shape[1]
        # TODO: this should sol(0)
        vp_init = np.zeros((dimredsys, 1))

    # instantiate the Time Int Parameters
    TsP = TimestepParams(methdict[method], N, scheme=scheme)

    if NtsList is not None:
        TsP.Ntslist = NtsList
    if LinaTol is not None:
        TsP.linatol = LinaTol
    if MaxIter is not None:
        TsP.MaxIter = MaxIter
    if tE is not None:
        TsP.tE = tE
    if Omega is not None:
        TsP.Omega = Omega
    if SaveTStps is not None:
        TsP.SaveTStps = SaveTStps
    if UsePreTStps is not None:
        TsP.UsePreTStps = UsePreTStps
    if SaveIniVal is not None:
        TsP.SaveIniVal = SaveIniVal
    if inikryupd is not None:
        TsP.inikryupd = inikryupd
    TsP.TolCorB = tolcor

    print 'Mesh parameter N = %d' % N
    print 'Time interval [%d,%1.2f]' % (TsP.t0, TsP.tE)
    print 'Omega = %d' % TsP.Omega
    print 'You have chosen %s for time integration' % methdict[method]
    print 'The tolerance for the linear solver is %e' % TsP.linatol
    print 'tolcor -- controlling the abs residuals -- is ', tolcor

    if method == 1:
        # Rearrange the matrices and rhs
        # from smamin_utils import col_columns_atend
        from scipy.io import loadmat

        if prob == 'cyl' and scheme == 'CR':
            if N == 0:
                cricell = 758
            elif N == 1:
                cricell = 1498
            elif N == 2:
                cricell = 2386
            elif N == 3:
                cricell = 4843
            else:
                raise NotImplementedError()
            # TODO: this is hard coded...
            # dptatnb = dolfin.Point(2.2, 0.2)
            # cricell = smt.get_cellid_nexttopoint(PrP.mesh, dptatnb)

        elif prob == 'cyl':
            raise NotImplementedError()
        else:
            cricell = None

        MSmeCL, ASmeCL, BSme, B2Inds, B2BoolInv, B2BI = smt.\
            get_smamin_rearrangement(N, PrP, M=Mc, A=Ac, B=Bc,
                                     crinicell=cricell, addnedgeat=cricell,
                                     scheme=scheme, fullB=Ba)

        FvbcSme = np.vstack([fvbc[~B2BoolInv, ], fvbc[B2BoolInv, ]])
        FpbcSme = fpbc

        # inivalue
        dname = 'IniValSmaMinN%s' % N
        try:
            IniV = loadmat(dname)
            qqpq_init = IniV['qqpq_old']
            vp_init = None
        except IOError:
            qqpq_init = None

    # Output
    try:
        os.chdir('json')
    except OSError:
        raise Warning('need "json" subdirectory for storing the data')
    os.chdir('..')

    if TsP.ParaviewOutput:
        os.chdir('results/')
        for fname in glob.glob(TsP.method + scheme + '*'):
            os.remove(fname)
        os.chdir('..')

    # ## Time stepping ## #
    for i, CurNTs in enumerate(TsP.Ntslist):
        TsP.Nts = CurNTs

        if method == 2:
            tis.halfexp_euler_nseind2(Mc, MPa, Ac, BTc, Bc, fvbc, fpbc,
                                      PrP, TsP, vp_init=vp_init)
        elif method == 1:
            tis.halfexp_euler_smarminex(MSmeCL, ASmeCL, BSme,
                                        MPa, FvbcSme, FpbcSme,
                                        B2BoolInv, PrP, TsP,
                                        qqpq_init=qqpq_init, vp_init=vp_init)

        # Output only in first iteration!
        TsP.ParaviewOutput = False

    save_simu(TsP, PrP)

    return


def plot_errs_res(TsP):

    plt.close('all')
    for i in range(len(TsP.Ntslist)):
        plt.figure(1)
        plt.plot(TsP.Residuals.ContiRes[i])
        plt.title('Lina residual in the continuity eqn')
        plt.figure(2)
        plt.plot(TsP.Residuals.VelEr[i])
        plt.title('Error in the velocity')
        plt.figure(3)
        plt.plot(TsP.Residuals.PEr[i])
        plt.title('Error in the pressure')

    plt.show(block=False)

    return


def plot_exactsolution(PrP, TsP):

    u_file = dolfin.File("results/exa_velocity.pvd")
    p_file = dolfin.File("results/exa_pressure.pvd")
    for tcur in np.linspace(TsP.t0, TsP.tE, 11):
        PrP.v.t = tcur
        PrP.p.t = tcur
        vcur = dolfin.project(PrP.v, PrP.V)
        pcur = dolfin.project(PrP.p, PrP.Q)
        u_file << vcur, tcur
        p_file << pcur, tcur


def save_simu(TsP, PrP, scheme=''):
    import json
    DictOfVals = {'SpaceDiscParam': PrP.N,
                  'Omega': PrP.omega,
                  'nu': PrP.nu,
                  'TimeInterval': [TsP.t0, TsP.tE],
                  'TimeDiscs': TsP.Ntslist,
                  'LinaTol': TsP.linatol,
                  'TimeIntMeth': TsP.method,
                  'ContiRes': TsP.Residuals.ContiRes,
                  'VelEr': TsP.Residuals.VelEr,
                  'PEr': TsP.Residuals.PEr,
                  'MomRes': TsP.Residuals.MomRes,
                  'DContiRes': TsP.Residuals.DContiRes,
                  'TolCor': TsP.TolCor}

    JsFile = 'json/Omeg%dTol%0.2eNTs%dto%dMesh%d' % (
        DictOfVals['Omega'],
        TsP.linatol,
        TsP.Ntslist[0],
        TsP.Ntslist[-1],
        PrP.N) + 'nu{0}'.format(PrP.nu) + TsP.method + scheme + '.json'

    f = open(JsFile, 'w')
    f.write(json.dumps(DictOfVals))

    print 'For the error plot, run\nimport plot_utils as ' +\
        'plu\nplu.jsd_plot_errs("' + JsFile + '")'
    print 'For the error valus, run\nimport plot_utils as ' +\
        'plu\nplu.jsd_calc_l2errs("' + JsFile + '")'

    return


class NseResiduals(object):

    def __init__(self):
        self.ContiRes = []
        self.VelEr = []
        self.PEr = []
        self.MomRes = []
        self.DContiRes = []


class UpFiles(object):

    def __init__(self, name=None, scheme=None):
        if name is not None:
            self.u_file = dolfin.File("results/{0}{1}".format(name, scheme) +
                                      "_velocity.pvd")
            self.p_file = dolfin.File("results/{0}{1}".format(name, scheme) +
                                      "_pressure.pvd")


if __name__ == '__main__':
    import dolfin_navier_scipy.data_output_utils as dou
    dou.logtofile(logstr='logfile_m1_cylinder_t12')

    scheme = 'CR'
    N = 3
    Re = 60
    tE = .2
    prob = 'cyl'
    tol = 2**(-18)
    Ntslist = [512]

    solve_euler_timedep(method=2, tE=tE, Re=Re, LinaTol=tol, tolcor=True,
                        MaxIter=800,
                        N=N, NtsList=Ntslist, scheme=scheme, prob=prob)

    # solve_euler_timedep(method=1, tE=1., LinaTol=2**(-12), tolcor=True,
    #                     MaxIter=400, N=40, NtsList=Ntslist)
