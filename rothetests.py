import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import sadptprj_riclyap_adi.lin_alg_utils as lau

dolfin.parameters.linear_algebra_backend = 'uBLAS'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2,
#                                              'convstatsl': [],
#                                              'maxiter': 200})
krylovdict = {}


def testit(problem='drivencavity', N=None, nu=None, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0, scheme=None):

    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problem, N=N, Re=Re,
                                                nu=nu, scheme=scheme,
                                                mergerhs=True)
    proutdir = 'results/'
    ddir = 'data/'
    data_prfx = problem + '_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE)

    M, A, J, = stokesmatsc['M'], stokesmatsc['A'], stokesmatsc['J']
    V, Q = femp['V'], femp['Q']
    invinds, diribcs = femp['invinds'],  femp['diribcs']
    fv, fp = rhsd['fv'], rhsd['fp']

    vp_iniv = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=-J.T,
                                   rhsv=fv, rhsp=fp)

    vini, pini = dts.expand_vp_dolfunc(V=V, Q=Q, vp=vp_iniv, invinds=invinds,
                                       diribcs=diribcs, ppin=-1)

    vfile = dolfin.File(proutdir + data_prfx + 'vfile.pvd'),
    pfile = dolfin.File(proutdir + data_prfx + 'pfile.pvd')
    vfile << vini
    pfile << pini

    vfunk = vini
    pfunk = pini

    get_rhs???

    # soldict = stokesmatsc  # containing A, J, JT
    # soldict.update(femp)  # adding V, Q, invinds, diribcs
    # soldict.update(tips)  # adding time integration params
    # soldict.update(rhsd)
    # soldict.update(N=N, nu=nu,
    #                vel_nwtn_stps=nnewtsteps,
    #                vel_nwtn_tol=vel_nwtn_tol,
    #                start_ssstokes=True,
    #                get_datastring=None,
    #                data_prfx=ddir+data_prfx,
    #                paraviewoutput=ParaviewOutput,
    #                vel_pcrd_stps=1,
    #                clearprvdata=True,
    #                vfileprfx=proutdir+'vel_{0}_'.format(scheme),
    #                pfileprfx=proutdir+'p_{0}_'.format(scheme))

    # soldict.update(krylovdict)  # if we wanna use an iterative solver

    # snu.solve_nse(**soldict)

if __name__ == '__main__':
    testit(problem='cylinderwake', N=3, Re=120, Nts=128, tE=.2,
           ParaviewOutput=True, scheme='CR')
