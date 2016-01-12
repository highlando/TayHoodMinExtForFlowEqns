import dolfin

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interpolationtest")
# disable the fenics loggers
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

fh = logging.FileHandler('log.interpoltest')
fh.setLevel(logging.DEBUG)

formatter = \
    logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
logger.addHandler(fh)

problemname = 'cylinderwake'
N = 2
Nref = 3
Re = 10

femp, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problemname, N=N, Re=Re,
                                            mergerhs=True)
soldict = stokesmatsc  # containing A, J, JT
soldict.update(femp)  # adding V, Q, invinds, diribcs
soldict.update(rhsd)  # adding the discrete rhs

v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(N=N, **soldict)

vwbc = dts.append_bcs_vec(v_ss_nse, **femp)
vwbcf, _ = dts.expand_vp_dolfunc(vc=vwbc, V=femp['V'])

fempref, stokesmatsc, rhsd = dnsps.get_sysmats(problem=problemname, N=Nref,
                                               Re=Re, mergerhs=True)
Vref = fempref['V']


class ExtVwbc(dolfin.Expression):
    def __init__(self, vfun=None):
        self.vfun = vfun

    def eval(self, value, x):
        try:
            self.vfun.eval(value, x)
        except RuntimeError:
            value[0] = 0.0
            value[1] = 0.0
            logger.info("got x={0}, gave value={1}".format(x, value))

    def value_shape(self):
        return (2,)

extv = ExtVwbc(vfun=vwbcf)
vwbcfinterp = dolfin.interpolate(extv, Vref)
dolfin.plot(vwbcfinterp, interactive=True)  # , mode="glyphs")
dolfin.plot(vwbcf, interactive=True)  # , mode="glyphs")

# vwbcfref = dolfin.interpolate(vwbcf, Vref)
# dolfin.plot(vwbcf, interactive=True)  # , mode="glyphs")
# vwbcfref = dolfin.interpolate(vwbcf, Vref)
