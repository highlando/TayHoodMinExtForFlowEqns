import dolfin
import numpy as np
from time_int_schemes import expand_vp_dolfunc, get_dtstr

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu

from prob_defs import FempToProbParams

import matlibplots.conv_plot_utils as cpu

dolfin.set_log_level(60)

samplerate = 2

N, Re, scheme, tE = 3, 60, 'CR', .2
Ntslist = [32, 64, 128, 256, 512]
Ntsref = 2048
tol = 0  # 2**(-18)

svdatapathref = 'data/'
svdatapath = 'data/'
# svdatapath = 'edithadata/'

femp, stokesmatsc, rhsd_vfrc, \
    rhsd_stbc, data_prfx, ddir, proutdir \
    = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                        scheme=scheme)
fpbc = rhsd_stbc['fp']

PrP = FempToProbParams(N, femp=femp, pdof=None)
dtstrdctref = dict(prefix=svdatapathref, method=2, N=PrP.N,
                   nu=PrP.nu, Nts=Ntsref, tol=0, te=tE)

J, MP = stokesmatsc['J'], stokesmatsc['MP']
Nv = J.shape[1]

method = 1

# get the ref trajectories
trange = np.linspace(0, tE, Ntsref+1)
M, A = stokesmatsc['M'], stokesmatsc['A']
JT, J = stokesmatsc['JT'], stokesmatsc['J']
invinds = femp['invinds']
fv, fp = rhsd_stbc['fv'], rhsd_stbc['fp']
ppin = None

snsedict = dict(A=A, J=J, JT=JT, M=M, ppin=ppin, fv=fv, fp=fp, V=femp['V'],
                invinds=invinds, diribcs=femp['diribcs'],
                start_ssstokes=True, trange=trange,
                return_dictofpstrs=True, return_dictofvelstrs=True)

vdref, pdref = snu.solve_nse(**snsedict)

errvl = []
errpl = []
rescl = []
for Nts in Ntslist:
    dtstrdct = dict(prefix=svdatapath, method=method, N=PrP.N,
                    nu=PrP.nu, Nts=Nts, tol=tol, te=tE)

    elv = []
    elp = []
    elc = []

    def app_pverr(tcur):
        cdatstr = get_dtstr(t=tcur, **dtstrdct)
        vp = np.load(cdatstr + '.npy')
        v, p = expand_vp_dolfunc(PrP, vp=vp)

        cdatstrref = get_dtstr(t=tcur, **dtstrdctref)
        vpref = np.load(cdatstrref + '.npy')
        vref, pref = expand_vp_dolfunc(PrP, vp=vpref)

        elv.append(dolfin.errornorm(v, vref))
        elp.append(dolfin.errornorm(p, pref))
        cres = J*vp[:Nv]-fpbc
        ncres = np.sqrt(np.dot(cres.T, MP*cres))[0][0]
        # routine from time_int_schemes seems buggy for CR or 'g not 0'
        # ncres = comp_cont_error(v, fpbc, PrP.Q)
        elc.append(ncres)

    trange = np.linspace(0, tE, Nts+1)
    samplvec = np.arange(1, len(trange), samplerate)

    app_pverr(0)

    for t in trange[samplvec]:
        app_pverr(t)

    ev = np.array(elv)
    ep = np.array(elp)
    ec = np.array(elc)

    trange = np.r_[trange[0], trange[samplerate]]
    dtvec = trange[1:] - trange[:-1]

    trapv = 0.5*(ev[:-1] + ev[1:])
    errv = (dtvec*trapv).sum()

    trapp = 0.5*(ep[:-1] + ep[1:])
    errp = (dtvec*trapp).sum()

    trapc = 0.5*(ec[:-1] + ec[1:])
    resc = (dtvec*trapc).sum()

    print 'Nts = {0}, v_error = {1}, p_error = {2}, contres={3}'.\
        format(Nts, errv, errp, resc)

    errvl.append(errv)
    errpl.append(errp)
    rescl.append(resc)

print errvl
print errpl

cpu.conv_plot(Ntslist, [errvl], logscale=2,
              markerl=['o'], fignum=1, leglist=['velerror'])
cpu.conv_plot(Ntslist, [rescl], logscale=2,
              markerl=['o'], fignum=2, leglist=['cres'])
cpu.conv_plot(Ntslist, [errpl],logscale=2,
              markerl=['o'], fignum=3, leglist=['perror'])
