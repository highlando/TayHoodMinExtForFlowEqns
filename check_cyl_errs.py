import dolfin
import numpy as np
from time_int_schemes import expand_vp_dolfunc, get_dtstr
import dolfin_navier_scipy.problem_setups as dnsps
from prob_defs import FempToProbParams

dolfin.set_log_level(60)

samplerate = 20

N, Re, scheme, tE = 3, 75, 'CR', 2.
Ntslist = [512, 1024, 2048]
Ntsref = 2*4096
tol = 0  # 2**(-12)

svdatapath = 'data/'

femp, stokesmatsc, rhsd_vfrc, \
    rhsd_stbc, data_prfx, ddir, proutdir \
    = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                        scheme=scheme)

PrP = FempToProbParams(N, femp=femp, pdof=None)
dtstrdctref = dict(prefix=svdatapath, method=2, N=PrP.N,
                   nu=PrP.nu, Nts=Ntsref, tol=0, te=tE)

method = 2

errvl = []
errpl = []
for Nts in Ntslist:
    dtstrdct = dict(prefix=svdatapath, method=method, N=PrP.N,
                    nu=PrP.nu, Nts=Nts, tol=tol, te=tE)

    elv = []
    elp = []

    def app_pverr(tcur):
        cdatstr = get_dtstr(t=tcur, **dtstrdct)
        vp = np.load(cdatstr + '.npy')
        v, p = expand_vp_dolfunc(PrP, vp=vp)

        cdatstrref = get_dtstr(t=tcur, **dtstrdctref)
        vpref = np.load(cdatstrref + '.npy')
        vref, pref = expand_vp_dolfunc(PrP, vp=vpref)

        elv.append(dolfin.errornorm(v, vref))
        elp.append(dolfin.errornorm(p, pref))

    trange = np.linspace(0, tE, Nts+1)
    samplvec = np.arange(1, len(trange), samplerate)

    app_pverr(0)

    for t in trange[samplvec]:
        app_pverr(t)

    ev = np.array(elv)
    ep = np.array(elp)

    trange = np.r_[trange[0], trange[samplerate]]
    dtvec = trange[1:] - trange[:-1]

    trapv = 0.5*(ev[:-1] + ev[1:])
    errv = (dtvec*trapv).sum()

    trapp = 0.5*(ep[:-1] + ep[1:])
    errp = (dtvec*trapp).sum()

    print 'Nts = {0}, v_error = {1}, p_error = {2}'.format(Nts, errv, errp)

    errvl.append(errv)
    errpl.append(errp)

print errvl
print errpl
