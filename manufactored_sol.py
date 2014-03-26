import dolfin
import numpy as np
import smartminex_tayhoomesh


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


class ProbParams(object):

    def __init__(self, N, Omega):

        self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
        self.N = N
        self.V = dolfin.VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = dolfin.FunctionSpace(self.mesh, "CG", 1)
        self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
        self.Pdof = 0  # dof removed in the p approximation
        self.omega = Omega
        self.nu = 0
        self.fp = dolfin.Constant((0))
        self.fv = dolfin.Expression(
            ("40*nu*pow(x[0],2)*pow(x[1],3)*sin(omega*t) - 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) + 24*nu*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*sin(omega*t) + 20*nu*pow(x[0],2)*x[1]*sin(omega*t) - 12*nu*pow(x[0],2)*pow((x[0] - 1),2)*sin(omega*t) - 32*nu*x[0]*pow(x[1],3)*sin(omega*t) + 48*nu*x[0]*pow(x[1],2)*sin(omega*t) - 16*nu*x[0]*x[1]*sin(omega*t) + 8*nu*pow(x[1],3)*pow((x[0] - 1),2)*sin(omega*t) - 12*nu*pow(x[1],2)*pow((x[0] - 1),2)*sin(omega*t) + 4*nu*x[1]*pow((x[0] - 1),2)*sin(omega*t) - 4*pow(x[0],3)*pow(x[1],2)*pow((x[0] - 1),3)*(2*x[0] - 1)*pow((x[1] - 1),2)*(2*x[1]*(x[1] - 1) + x[1]*(2*x[1] - 1) + (x[1] - 1)*(2*x[1] - 1) - 2*pow((2*x[1] - 1),2))*pow(sin(omega*t),2) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*omega*cos(omega*t) + 6*pow(x[0],2)*pow(x[1],2)*pow((x[0] - 1),2)*omega*cos(omega*t) - 2*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*omega*cos(omega*t) + 2*x[0]*pow(x[1],2)*sin(omega*t) - 2*x[0]*x[1]*sin(omega*t) - pow(x[1],2)*sin(omega*t) + x[1]*sin(omega*t)",
             "-40*nu*pow(x[0],3)*pow(x[1],2)*sin(omega*t) + 32*nu*pow(x[0],3)*x[1]*sin(omega*t) - 8*nu*pow(x[0],3)*pow((x[1] - 1),2)*sin(omega*t) + 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) - 48*nu*pow(x[0],2)*x[1]*sin(omega*t) + 12*nu*pow(x[0],2)*pow((x[1] - 1),2)*sin(omega*t) - 24*nu*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) - 20*nu*x[0]*pow(x[1],2)*sin(omega*t) + 16*nu*x[0]*x[1]*sin(omega*t) - 4*nu*x[0]*pow((x[1] - 1),2)*sin(omega*t) + 12*nu*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) + 4*pow(x[0],3)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*pow((x[1] - 1),3)*(2*x[1] - 1)*(2*x[0]*(x[0] - 1) + x[0]*(2*x[0] - 1) + (x[0] - 1)*(2*x[0] - 1) - 2*pow((2*x[0] - 1),2))*pow(sin(omega*t),2) - 6*pow(x[0],2)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) + 2*pow(x[0],2)*x[1]*sin(omega*t) - pow(x[0],2)*sin(omega*t) + 2*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 2*x[0]*x[1]*sin(omega*t) + x[0]*sin(omega*t)"),
            t=0,
            nu=self.nu,
            omega=self.omega)

        self.v = dolfin.Expression((
            "sin(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
            "sin(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega=self.omega, t=0)
        self.vdot = dolfin.Expression((
            "omega*cos(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
            "omega*cos(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega=self.omega, t=0)
        self.p = dolfin.Expression(
            ("sin(omega*t)*x[0]*(1-x[0])*x[1]*(1-x[1])"),
            omega=self.omega,
            t=0)

        bcinds = []
        for bc in self.velbcs:
            bcdict = bc.get_boundary_values()
            bcinds.extend(bcdict.keys())

        # indices of the inner velocity nodes
        self.invinds = np.setdiff1d(range(self.V.dim()), bcinds)
