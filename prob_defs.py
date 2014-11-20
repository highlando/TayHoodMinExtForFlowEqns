import dolfin
import numpy as np
import smartminex_tayhoomesh
import sympy as smp

from dolfin import Expression


def setget_velbcs_zerosq(mesh, V):
    # Boundaries
    def top(x, on_boundary):
        return np.fabs(x[1] - 1.0) < dolfin.DOLFIN_EPS
        # and (np.fabs(x[0]) > DOLFIN_EPS))
        # and np.fabs(x[0] - 1.0) > DOLFIN_EPS )

    def leftbotright(x, on_boundary):
        return (np.fabs(x[0] - 1.0) < dolfin.DOLFIN_EPS
                or np.fabs(x[1]) < dolfin.DOLFIN_EPS
                or np.fabs(x[0]) < dolfin.DOLFIN_EPS)

    # No-slip boundary condition for velocity
    noslip = dolfin.Constant((0.0, 0.0))
    bc0 = dolfin.DirichletBC(V, noslip, leftbotright)

    # Boundary condition for velocity at the lid
    lid = dolfin.Constant((0.0, 0.0))
    bc1 = dolfin.DirichletBC(V, lid, top)

    # Collect boundary conditions
    velbcs = [bc0, bc1]

    return velbcs


def comp_symb_nserhs(u=None, v=None, p=None, nu=None):
    from sympy import diff

    # space and time variables
    x, y, t = smp.symbols('x[0], x[1], t')
    # Stokes case
    rhs1 = smp.simplify(
        diff(u, t) - nu * smp.simplify(diff(u, x, x) + diff(u, y, y)) +
        diff(p, x))
    rhs2 = smp.simplify(
        diff(v, t) - nu * smp.simplify(diff(v, x, x) + diff(v, y, y)) +
        diff(p, y))

    # rhs3 = div u --- should be zero!!
    rhs3 = diff(u, x) + diff(v, y)

    # Advection (u.D)u
    ad1 = smp.simplify(smp.simplify(u * diff(u, x)) +
                       smp.simplify(v * diff(u, y)))
    ad2 = smp.simplify(smp.simplify(u * diff(v, x)) +
                       smp.simplify(v * diff(v, y)))

    rhs1 = smp.simplify(rhs1 + ad1)
    rhs2 = smp.simplify(rhs2 + ad2)
    rhs3 = smp.simplify(rhs3)

    return rhs1, rhs2, rhs3


class ProbParams(object):

    def __init__(self, N, omega=None, nu=None, scheme='TH'):

        self.N = N
        if scheme == 'TH':
            self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
            self.V = dolfin.VectorFunctionSpace(self.mesh, "CG", 2)
            self.Q = dolfin.FunctionSpace(self.mesh, "CG", 1)
        elif scheme == 'CR':
            self.mesh = dolfin.UnitSquareMesh(N, N)
            self.V = dolfin.VectorFunctionSpace(self.mesh, "CR", 1)
            self.Q = dolfin.FunctionSpace(self.mesh, "DG", 0)
        self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
        self.Pdof = 0  # dof removed in the p approximation
        self.omega = omega
        self.nu = nu

        x, y, t, nu, om = smp.symbols('x[0], x[1], t, nu, omega')

        ft = smp.sin(om*t)

        u1 = ft*x*x*(1 - x)*(1 - x)*2*y*(1 - y)*(2*y - 1)
        u2 = ft*y*y*(1 - y)*(1 - y)*2*x*(1 - x)*(1 - 2*x)
        p = ft*x*(1 - x)*y*(1 - y)

        du1 = smp.diff(u1, t)
        du2 = smp.diff(u2, t)

        rhs1, rhs2, rhs3 = comp_symb_nserhs(u=u1, v=u2, p=p, nu=self.nu)

        from sympy.printing import ccode
        self.v = Expression((ccode(u1), ccode(u2)),
                            t=0.0, omega=self.omega)
        self.p = Expression((ccode(p)),
                            t=0.0, omega=self.omega)
        self.fv = Expression((ccode(rhs1), ccode(rhs2)),
                             t=0.0, omega=self.omega, nu=self.nu)
        self.fp = Expression((ccode(rhs3)),
                             t=0.0, omega=self.omega)
        self.vdot = Expression((ccode(du1), ccode(du2)),
                               t=0.0, omega=self.omega)

        bcinds = []
        for bc in self.velbcs:
            bcdict = bc.get_boundary_values()
            bcinds.extend(bcdict.keys())

        # indices of the inner velocity nodes
        self.invinds = np.setdiff1d(range(self.V.dim()), bcinds)
