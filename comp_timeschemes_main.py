import dolfin
# import plot_utils as plu

import numpy as np
# import scipy.sparse as sps
import matplotlib.pyplot as plt
import os
import glob

import dolfin_to_nparrays as dtn
import time_int_schemes as tis
import smartminex_tayhoomesh

from prob_defs import ProbParams


class TimestepParams(object):

    def __init__(self, method, N):
        self.t0 = 0
        self.tE = 1.0
        self.Omega = 8
        self.Ntslist = [128]
        self.NOutPutPts = 16
        self.method = method
        self.SadPtPrec = True
        self.UpFiles = UpFiles(method)
        self.Residuals = NseResiduals()
        self.globalcounts = GlobalCounts()
        self.linatol = 1e-4  # 0 for direct sparse solver
        self.TolCor = []
        self.MaxIter = 85
        self.Ml = None  # preconditioners
        self.Mr = None
        self.ParaviewOutput = False
        self.SaveIniVal = False
        self.SaveTStps = False
        self.UsePreTStps = False
        self.TolCorB = True


def solve_euler_timedep(method=1, Omega=8, tE=None, Prec=None,
                        N=40, NtsList=None, LinaTol=None, MaxIter=None,
                        UsePreTStps=None, SaveTStps=None, SaveIniVal=None,
                        krylovini=None, globalcount=False):
    """system to solve

             du\dt + (u*D)u + grad p = fv
                      div u          = fp

    """

    methdict = {
        1: 'HalfExpEulSmaMin',
        2: 'HalfExpEulInd2'}

    # instantiate object containing mesh, V, Q, rhs, velbcs, invinds
    # set nu=0 for Euler flow
    PrP = ProbParams(N, omega=Omega, nu=0)
    # instantiate the Time Int Parameters
    TsP = TimestepParams(methdict[method], N)

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

    print 'Mesh parameter N = %d' % N
    print 'Time interval [%d,%1.2f]' % (TsP.t0, TsP.tE)
    print 'Omega = %d' % TsP.Omega
    print 'You have chosen %s for time integration' % methdict[method]
    print 'The tolerance for the linear solver is %e' % TsP.linatol

    # get system matrices as np.arrays
    Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q)
    fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

    # condense the system by resolving the boundary values
    (Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals,
     invinds) = dtn.condense_sysmatsbybcs(Ma, Aa, BTa, Ba, fv, fp, PrP.velbcs)

    if method == 1:
        # Rearrange the matrices and rhs
        # from smamin_utils import col_columns_atend
        from scipy.io import loadmat

        MSmeCL, BSme, B2Inds, B2BoolInv, B2BI = smartminex_tayhoomesh.\
            get_smamin_rearrangement(N, PrP, Mc, Bc)

        FvbcSme = np.vstack([fvbc[~B2BoolInv, ], fvbc[B2BoolInv, ]])
        FpbcSme = fpbc

        PrP.Pdof = 0  # Thats how the smamin is constructed

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
        for fname in glob.glob(TsP.method + '*'):
            os.remove(fname)
        os.chdir('..')

    #
    # Time stepping
    #
    # starting value
    dimredsys = len(fvbc) + len(fp) - 1
    vp_init = np.zeros((dimredsys, 1))

    for i, CurNTs in enumerate(TsP.Ntslist):
        TsP.Nts = CurNTs

        if method == 2:
            tis.halfexp_euler_nseind2(Mc, MPa, Ac, BTc, Bc, fvbc, fpbc,
                                      vp_init, PrP, TsP,
                                      krylovini=krylovini,
                                      globalcount=globalcount)
        elif method == 1:
            tis.halfexp_euler_smarminex(MSmeCL, BSme, MPa, FvbcSme, FpbcSme,
                                        B2BoolInv, PrP, TsP, vp_init,
                                        qqpq_init=qqpq_init,
                                        krylovini=krylovini,
                                        globalcount=globalcount)

        # Output only in first iteration!
        TsP.ParaviewOutput = False

    save_simu(TsP, PrP, globalcount=globalcount, krylovini=krylovini)

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


def save_simu(TsP, PrP, globalcount=False, krylovini=None):
    import json
    DictOfVals = {'SpaceDiscParam': PrP.N,
                  'Omega': PrP.omega,
                  'TimeInterval': [TsP.t0, TsP.tE],
                  'TimeDiscs': TsP.Ntslist,
                  'LinaTol': TsP.linatol,
                  'TimeIntMeth': TsP.method,
                  'ContiRes': TsP.Residuals.ContiRes,
                  'VelEr': TsP.Residuals.VelEr,
                  'PEr': TsP.Residuals.PEr,
                  'TolCor': TsP.TolCor}

    JsFile = 'json/Omeg%dTol%0.2eNTs%dto%dMesh%d' % (
        DictOfVals['Omega'],
        TsP.linatol,
        TsP.Ntslist[0],
        TsP.Ntslist[-1],
        PrP.N) + TsP.method + '.json'

    if globalcount:
        DictOfVals.update({'NumIter': TsP.globalcounts.NumIter,
                           'TimeIter': TsP.globalcounts.TimeIter})
        JsFile = JsFile + '_globalcount' + '_kiniv{0}'.format(krylovini)
        f = open(JsFile, 'w')
        f.write(json.dumps(DictOfVals))
        print 'For the time/iter counts run\nimport plot_utils as ' +\
            'plu\nplu.jsd_count_timeiters("' + JsFile + '")'

        return

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


class GlobalCounts(object):

    def __init__(self):
        self.TimeIter = []
        self.NumIter = []
        self.Info = ''


class UpFiles(object):

    def __init__(self, name=None):
        if name is not None:
            self.u_file = dolfin.File("results/%s_velocity.pvd" % name)
            self.p_file = dolfin.File("results/%s_pressure.pvd" % name)
        else:
            self.u_file = dolfin.File("results/velocity.pvd")
            self.p_file = dolfin.File("results/pressure.pvd")

if __name__ == '__main__':
    import dolfin_navier_scipy.data_output_utils as dou
    dou.logtofile(logstr='logfile4')
    # solve_euler_timedep(method=1, N=40, LinaTol=2**(-12),
    #                     MaxIter=85, NtsList=[512])
    Ntsl = [16, 32]  # , 64, 128, 256]
    method = 1
    # solve_euler_timedep(method=method, N=40, LinaTol=2**(-12),
    #                     MaxIter=800, NtsList=Ntsl, globalcount=True,
    #                     krylovini='upd')
    # solve_euler_timedep(method=method, N=40, LinaTol=2**(-12),
    #                     MaxIter=800, NtsList=Ntsl, globalcount=True,
    #                     krylovini='old')
    # solve_euler_timedep(method=method, N=40, LinaTol=2**(-12),
    #                     MaxIter=800, NtsList=Ntsl, globalcount=True,
    #                     krylovini='zero')
    method = 2
    solve_euler_timedep(method=method, N=10, LinaTol=2**(-12),
                        MaxIter=800, NtsList=Ntsl, globalcount=True,
                        krylovini='upd')
    # solve_euler_timedep(method=method, N=40, LinaTol=2**(-12),
    #                     MaxIter=800, NtsList=Ntsl, globalcount=True,
    #                     krylovini='old')
    # solve_euler_timedep(method=method, N=40, LinaTol=2**(-12),
    #                     MaxIter=800, NtsList=Ntsl, globalcount=True,
    #                     krylovini='zero')
