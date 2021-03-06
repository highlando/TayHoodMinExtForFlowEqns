import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import matlibplots.conv_plot_utils as cpu

import rothe_utils as rtu
from time_int_schemes import get_dtstr

dolfin.parameters.linear_algebra_backend = 'uBLAS'

import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rothemain")
# disable the fenics loggers
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
logger.setLevel('INFO')  # default is DEBUG

fh = logging.FileHandler('log.rothemain')
# fh.setLevel(logging.DEBUG)
fh.setLevel(logging.INFO)

formatter = \
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.Formatter('%(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
logger.addHandler(fh)

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

proutdir = 'results/'
ddir = 'data/'

debug = True
debug = False


def check_the_sim(problem='cylinderwake', nswtchsl=[2], scheme='CR',
                  nu=None, Re=None, Nts=None, paraout=False, t0=None, tE=None,
                  plotit=False, dtstrdct={}, debug=False, Nref=3,
                  method=2, refmethod=2):
    trange = np.linspace(t0, tE, Nts+1)
    refdtstrdct = dict(prefix=ddir+problem+scheme,
                       method=refmethod, N=None, Nts=Nts, t0=t0, te=tE)
    refdatstr = get_dtstr(t=None, **refdtstrdct)
    compargs = dict(problem=problem, N=Nref, nu=nu, Re=Re, trange=trange,
                    scheme=scheme, data_prefix=refdatstr, debug=debug)
    refvdict, refpdict = dou.load_or_comp(filestr=[refdatstr+'_refvdict',
                                                   refdatstr+'_refpdict'],
                                          comprtn=gettheref,
                                          comprtnargs=compargs,
                                          debug=False, itsadict=True)

    viniv = dou.load_npa(refvdict['{0}'.format(t0)])
    piniv = dou.load_npa(refpdict['{0}'.format(t0)])

    # set up the list of the mesh parameters at every time step
    swl = Nts/len(nswtchsl)
    Nll = np.array([nswtchsl[0]]*(Nts+1))
    for k, N in enumerate(nswtchsl[1:]):
        Nll[(k+1)*swl+1:] = N
    Nll[0] = Nref
    Nll = Nll.tolist()
    nswtchstr = 'Nswitches' + ''.join(str(e) for e in nswtchsl)
    dtstrdct['prefix'] = dtstrdct['prefix'] + nswtchstr
    dtstrdct.update(t=None)
    cdatstr = get_dtstr(**dtstrdct)

    compvpdargs = dict(problem=problem, scheme=scheme, Re=Re, nu=nu,
                       t0=t0, tE=tE, Nts=Nts, viniv=viniv, piniv=piniv,
                       Nini=Nref, Nll=Nll, dtstrdct=dtstrdct, method=method)
    logger.info('try to load `' + cdatstr + '_vdict`, `' + cdatstr + '_pdict`')
    vdict, pdict = dou.load_or_comp(filestr=[cdatstr+'_vdict',
                                             cdatstr+'_pdict'],
                                    comprtn=rtu.rothe_time_int,
                                    comprtnargs=compvpdargs,
                                    debug=debug, itsadict=True)

    # vdict, pdict = rtu.rothe_ind2(problem=problem, scheme=scheme,
    #                               Re=Re, nu=nu,
    #                               t0=t0, tE=tE, Nts=Nts,
    #                               viniv=viniv, piniv=piniv, Nini=Nref,
    #                               Nll=Nll, dtstrdct=dtstrdct)

    compvperrargs = dict(problem=problem, scheme=scheme,
                         trange=trange.tolist(),
                         rvd=refvdict, rpd=refpdict, Nref=Nref,
                         cvd=vdict, cpd=pdict, Nll=Nll)

    errdict = dou.load_or_comp(filestr=[cdatstr+'_vperrdict'],
                               comprtn=compvperr, debug=debug,
                               comprtnargs=compvperrargs, itsadict=True)

    if plotit:
        rtu.plottimeerrs(trange=trange, perrl=[errdict['perrl']],
                         verrl=[errdict['verrl']], showplot=True)

    return (errdict['verrl'], errdict['perrl'],
            errdict['verrint'], errdict['perrint'], trange.tolist())


def compvperr(problem=None, scheme=None, trange=None,
              rvd=None, rpd=None, Nref=None,
              cvd=None, cpd=None, Nll=None):
    verrl, perrl = [], []

    rmd = rtu.get_curmeshdict(problem=problem, N=Nref,
                              scheme=scheme, onlymesh=True)
    for tk, t in enumerate(trange):
        vref = dou.load_npa(rvd['{0}'.format(t)])
        pref = dou.load_npa(rpd['{0}'.format(t)])
        # vreff, preff = dts.\
        #     expand_vp_dolfunc(vc=vref, V=rmd['V'], pc=pref, Q=rmd['Q'])

        cmd = rtu.get_curmeshdict(problem=problem, N=Nll[tk],
                                  scheme=scheme, onlymesh=True)
        try:
            vcur = dou.load_npa(cvd[t])
            pcur = dou.load_npa(cpd[t])
        except KeyError:
            vcur = dou.load_npa(cvd['{0}'.format(t)])
            pcur = dou.load_npa(cpd['{0}'.format(t)])

        logger = logging.getLogger("rothemain.compvperr")
        logger.debug("len v={0}, dim V={1}".format(vcur.size, cmd['V'].dim()))
        vcurf, pcurf = dts.\
            expand_vp_dolfunc(vc=vcur, V=cmd['V'], pc=pcur, Q=cmd['Q'])

        # # # This would be the FEniCS way to compute the diff
        #
        # verrl.append(dolfin.errornorm(vreff, vcurf, degree_rise=2))
        #
        # # # however this gave strange results (due to rounding errs?)

        # # # instead we interpolate and substract manually
        Vref = rmd['V']
        vcfinvref = dolfin.interpolate(vcurf, Vref)
        vcinvrefvec = vcfinvref.vector().array()
        difffunc = dolfin.Function(Vref)
        difffunc.vector().set_local(vref.flatten()-vcinvrefvec)

        Qref = rmd['Q']
        pcfinvref = dolfin.interpolate(pcurf, Qref)
        pcinvrefvec = pcfinvref.vector().array()
        difffuncp = dolfin.Function(Qref)
        difffuncp.vector().set_local(pref.flatten()-pcinvrefvec)

        verrl.append(dolfin.norm(difffunc))
        perrl.append(dolfin.norm(difffuncp))

    pverr, pperr, pt = verrl[0], perrl[0], trange[0]
    verrint, perrint = 0, 0
    for k in range(len(trange)-1):
        cverr, cperr, ct = verrl[k+1], perrl[k+1], trange[k+1]
        cts = ct - pt
        logger.debug('t={0}, cts={3}, cverr={1}, verrint={2}'.
                     format(ct, cverr, verrint, cts))
        verrint += 0.5*cts*(pverr + cverr)
        perrint += 0.5*cts*(pperr + cperr)
        pverr, pperr, pt = cverr, cperr, ct

    return dict(verrl=verrl, perrl=perrl, trange=trange,
                perrint=perrint, verrint=verrint,
                Nll=Nll, Nref=Nref)


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
    Re = 60
    Nref = 3
    Ntslist = [2048, 4096]  # [16]
    methlist = [1, 2]  # {1: 'smaminex', 2: 'ind2'}
    t0, tE = 0.0, 2.0
    nswtchshortlist = [3, 2, 3]  # , 2, 3]  # we recommend to start with `Nref`

    verrl, perrl, tmeshl = [], [], []
    legl = []
    keeppointslist = []
    for method in methlist:
        for Nts in Ntslist:
            dtstrdct = dict(prefix=ddir+problem+scheme, Nref=Nref,
                            method=method, N=None, Nts=Nts, t0=t0, te=tE)
            verr, perr, perrint, verrint, tmesh = \
                check_the_sim(problem='cylinderwake', method=method,
                              nswtchsl=nswtchshortlist, scheme='CR', Re=Re,
                              Nts=Nts, paraout=False, t0=t0, tE=tE,
                              dtstrdct=dtstrdct, debug=debug)
            tmeshl.append(tmesh[1:])
            verrl.append(verr[1:])
            perrl.append(perr[1:])
            nswtchstr = 'N' + ''.join(str(e) for e in nswtchshortlist)
            legl.append('$\\mathsf{{Nts}}={0}$ (index-{1})'.
                        format(Nts, method))
            logger.info(nswtchstr + ': Nts={0}, m={1}, veint={2}, peint={3}'.
                        format(Nts, method, verrint, perrint))
            curptkl = []
            for k in np.arange(1, len(nswtchshortlist)):
                jumpindx = k*(len(tmesh)-1)/len(nswtchshortlist)  # an int !!
                curptkl.extend([jumpindx-1, jumpindx])
            keeppointslist.append(curptkl)
    # markerl = ['s', '^', '.', '*', 'o', '+']
    xticks = np.linspace(t0, tE, 5)
    ppltdict = dict(leglist=legl, xlabel='t', xticks=xticks, logscaley=2,
                    usedefaultmarkers=True)

    cpu.para_plot(None, perrl, abscissal=tmeshl, fignum=11,
                  title='pointwise error in $p$',
                  tikzfile='p-pointwiseerror.tex',
                  xlims=[t0, tE],
                  downsample=50, keeppointslist=keeppointslist,
                  **ppltdict)
    cpu.para_plot(None, verrl, abscissal=tmeshl, fignum=22,
                  title='pointwise error in $q$',
                  tikzfile='vel-pointwiseerror.tex',
                  xlims=[t0, tE],
                  downsample=50,
                  **ppltdict)
