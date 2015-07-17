from dolfin import *					# /bin/bash !!!
import numpy.linalg as npla

import numpy as np
import os
import glob

import dolfin_to_nparrays as dtn
# import time_int_schemes as tis
# import smartminex_tayhoomesh

dolfin.parameters.linear_algebra_backend = "uBLAS"
# # # # # # # # # # # # # # #


# Function to solve the Stokes/Euler equations
#
def solve_stokesTimeDep(method=None, Omega=None, tE=None,
                        Prec=None, N=None, NtsList=None,
                        LinaTol=None, MaxIter=None, UsePreTStps=None,
                        SaveTStps=None, SaveIniVal=None):
    # set parameters
    if N is None:
        N = 4  # 12
    if method is None:
        method = 1 				# half explicit, our algorithm
    if Omega is None:
        Omega = 8
    methdict = {1: 'HalfExpEulSmaMin', 2: 'HalfExpEulInd2'}

    # instantiate object containing mesh, V, Q, rhs, velbcs, invinds
    PrP = ProbParams(N, Omega)
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
    Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
        dtn.condense_sysmatsbybcs(Ma, Aa, BTa, Ba, fv, fp, PrP.velbcs)

    if method != 2:					# Wenn Minimal Extension
        # Rearrange the matrices and rhs
        from scipy.io import loadmat


# AB HIER MUSS MAN ANPASSEN
        MSmeCL, BSme, B2Inds, B2BoolInv, B2BI = \
            smartminex_CRmesh.get_smamin_rearrangement(N, PrP, Mc, Bc)

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

    ###
    # Time stepping
    ###
    # starting value
    dimredsys = len(fvbc) + len(fp) - 1
    vp_init = np.zeros((dimredsys, 1))

    for i, CurNTs in enumerate(TsP.Ntslist):
        TsP.Nts = CurNTs

        if method == 2:
            tis.halfexp_euler_nseind2(Mc, MPa, Ac, BTc, Bc, fvbc, fpbc,
                                      vp_init, PrP, TsP)
        elif method == 1:
            tis.halfexp_euler_smarminex(MSmeCL, BSme, MPa, FvbcSme, FpbcSme,
                                        B2BoolInv, PrP, TsP, vp_init,
                                        qqpq_init=qqpq_init)

        # Output only in first iteration!
        TsP.ParaviewOutput = False

    JsD = save_simu(TsP, PrP)

    return


# Set Parameters for time integration such as
#   - t_0, t_f
#   - tol, max nr of steps
class TimestepParams(object):

    def __init__(self, method, N):
        self.t0 = 0
        self.tE = 1.0
        self.Omega = 8
        self.Ntslist = [32]
        self.NOutPutPts = 32
        self.method = method
        self.SadPtPrec = True
        self.UpFiles = UpFiles(method)
        self.Residuals = NseResiduals()
        self.linatol = 1e-3  # 0 for direct sparse solver
        self.TolCor = []
        self.MaxIter = 200
        self.Ml = None  # preconditioners
        self.Mr = None
        self.ParaviewOutput = False
        self.SaveIniVal = False
        self.SaveTStps = False
        self.UsePreTStps = False
        self.TolCorB = True


# Define Problem Parameters such as
#  - the mesh
#  - exact solution and nu, omega
#  - Ansatz spaces V, Q
class ProbParams(object):

    def __init__(self, N, Omega):
        self.mesh = UnitSquareMesh(
            N,
            N)  # smartminex_tayhoomesh.getmake_mesh(N)
        self.N = N
        self.V = VectorFunctionSpace(
            self.mesh,
            "CR",
            1)		# CR-P0 oder VectorFunctionSpace
        self.Q = FunctionSpace(
            self.mesh,
            "DG",
            0)			# nur scalar (daher ohne Vector)
        self.velbcs = setget_velbcs_zerosq(
            self.mesh,
            self.V)		# define Dirichlet boundary
        self.Pdof = 0  							# dof removed in the p approximation
        self.omega = Omega
        self.nu = 0							# nu=0 also Euler Gleichung
        self.fp = Constant((0))

        self.fv = Expression(
            ("40*nu*pow(x[0],2)*pow(x[1],3)*sin(omega*t) - 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) + 24*nu*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*sin(omega*t) + 20*nu*pow(x[0],2)*x[1]*sin(omega*t) - 12*nu*pow(x[0],2)*pow((x[0] - 1),2)*sin(omega*t) - 32*nu*x[0]*pow(x[1],3)*sin(omega*t) + 48*nu*x[0]*pow(x[1],2)*sin(omega*t) - 16*nu*x[0]*x[1]*sin(omega*t) + 8*nu*pow(x[1],3)*pow((x[0] - 1),2)*sin(omega*t) - 12*nu*pow(x[1],2)*pow((x[0] - 1),2)*sin(omega*t) + 4*nu*x[1]*pow((x[0] - 1),2)*sin(omega*t) - 4*pow(x[0],3)*pow(x[1],2)*pow((x[0] - 1),3)*(2*x[0] - 1)*pow((x[1] - 1),2)*(2*x[1]*(x[1] - 1) + x[1]*(2*x[1] - 1) + (x[1] - 1)*(2*x[1] - 1) - 2*pow((2*x[1] - 1),2))*pow(sin(omega*t),2) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*omega*cos(omega*t) + 6*pow(x[0],2)*pow(x[1],2)*pow((x[0] - 1),2)*omega*cos(omega*t) - 2*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*omega*cos(omega*t) + 2*x[0]*pow(x[1],2)*sin(omega*t) - 2*x[0]*x[1]*sin(omega*t) - pow(x[1],2)*sin(omega*t) + x[1]*sin(omega*t)",
             "-40*nu*pow(x[0],3)*pow(x[1],2)*sin(omega*t) + 32*nu*pow(x[0],3)*x[1]*sin(omega*t) - 8*nu*pow(x[0],3)*pow((x[1] - 1),2)*sin(omega*t) + 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) - 48*nu*pow(x[0],2)*x[1]*sin(omega*t) + 12*nu*pow(x[0],2)*pow((x[1] - 1),2)*sin(omega*t) - 24*nu*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) - 20*nu*x[0]*pow(x[1],2)*sin(omega*t) + 16*nu*x[0]*x[1]*sin(omega*t) - 4*nu*x[0]*pow((x[1] - 1),2)*sin(omega*t) + 12*nu*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) + 4*pow(x[0],3)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*pow((x[1] - 1),3)*(2*x[1] - 1)*(2*x[0]*(x[0] - 1) + x[0]*(2*x[0] - 1) + (x[0] - 1)*(2*x[0] - 1) - 2*pow((2*x[0] - 1),2))*pow(sin(omega*t),2) - 6*pow(x[0],2)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) + 2*pow(x[0],2)*x[1]*sin(omega*t) - pow(x[0],2)*sin(omega*t) + 2*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 2*x[0]*x[1]*sin(omega*t) + x[0]*sin(omega*t)"),
            t=0,
            nu=self.nu,
            omega=self.omega)

        self.v = Expression((
            "sin(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
            "sin(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega=self.omega, t=0)
        self.vdot = Expression((
            "omega*cos(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
            "omega*cos(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega=self.omega, t=0)
        self.p = Expression(
            ("sin(omega*t)*x[0]*(1-x[0])*x[1]*(1-x[1])"),
            omega=self.omega,
            t=0)

        bcinds = []
        for bc in self.velbcs:
            # dictionary fuer boundary(knoten_ind-value)
            bcdict = bc.get_boundary_values()
            bcinds.extend(bcdict.keys())			# nur die indices in Liste speichern

        # indices of the inner velocity nodes
        self.invinds = np.setdiff1d(
            range(
                self.V.dim()),
            bcinds)  # indices der inneren Knoten

# Definition of Dirichlet boundary
#  velbcs = boundary conditions for velocity (2 boundary parts)


def setget_velbcs_zerosq(mesh, V):
    # Boundaries
    def top(x, on_boundary):
        return np.fabs(x[1] - 1.0) < DOLFIN_EPS
        # and (np.fabs(x[0]) > DOLFIN_EPS))
        # and np.fabs(x[0] - 1.0) > DOLFIN_EPS )

    def leftbotright(x, on_boundary):
        return (np.fabs(x[0] - 1.0) < DOLFIN_EPS
                or np.fabs(x[1]) < DOLFIN_EPS
                or np.fabs(x[0]) < DOLFIN_EPS)

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(V, noslip, leftbotright)

    # Boundary condition for velocity at the lid
    lid = Constant((0.0, 0.0))
    bc1 = DirichletBC(V, lid, top)

    # Collect boundary conditions
    velbcs = [bc0, bc1]

    return velbcs



# FERTIG!
# returns the DoF corresponding to the space V_2h in the Preprint
# returns a vector with the indices!
# koennte man vermutlich auch eleganter ohne die Uebergabe der B Matrix machen
def get_SmartDoFs_CR(V, mesh, B_matrix):
    # apply algorithm from Preprint
    edges_V2 = computeSmartMinExtMapping(PrP.V, PrP.mesh)
    # get corresponding degrees of freedom of the CR-scheme
    print 'corresponding DoF for CR'
    edgeCRDofArray = computeEdgeCRDofArray(PrP.V, PrP.mesh)
    DoF_for_V2 = edgeCRDofArray[edges_V2.astype(int)].astype(int)
    DoF_for_V2_x = DoF_for_V2[:, 0]
    DoF_for_V2_y = DoF_for_V2[:, 1]
    # we still have do decide which of the two basis functions
    # corresponding to the edge we take
    # here: take as default [phi_E; 0] if not div[phi_E; 0] = 0
    # (check the column in B_matrix)
    # ??? ist pressure-DoF von B_matrix schon entfernt?
    dof_for_regular_B2 = DoF_for_V2_x
    for i in np.arange(len(edges_V2)):
        # take x-DoF and test whether its a zero-column
        dof = DoF_for_V2_x[i]
        col = Ba[:, dof]
        if npla.norm(col.toarray(), np.inf) < 1e-13:
            # norm to small -> seems to be a zero-column
            print 'nimm y'
            dof_for_regular_B2[i] = DoF_for_V2_y[i]
        else:
            print 'nimm x'

    return dof_for_regular_B2


# START Simulation
if __name__ == '__main__':
    #  solve_stokesTimeDep()
    PrP = ProbParams(2, 8)
    Vsmooth = VectorFunctionSpace(PrP.mesh, "Lagrange", 1)
    u = Function(PrP.V)
    p = Function(PrP.Q)

    # print 'NumOfNodes %3g and dimension %3g' % (PrP.mesh.num_vertices(),
    # len(u_array))
    print ' ------------ '
    print 'Mesh parameter N = %d' % PrP.N
    print 'Number of Triangles  %4g' % (PrP.mesh.num_cells())
    print 'Number of Nodes      %4g' % (PrP.mesh.num_vertices())
    print 'Number of Edges      %4g' % (PrP.mesh.num_edges())
    print 'Number of Facets     %4g' % (PrP.mesh.num_facets())
    print ' ------------ '

    c4n = PrP.mesh.coordinates()
    n4e = PrP.mesh.cells()

    u_array = u.vector().array()
    p_array = p.vector().array()

    print 'Dimensions: u has dim %3g and p has dim %3g' \
        % (len(u_array), len(p_array))

    # compute matrices
    Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q)
    # delete row corresponding to pressure-DoF (T_0)
    Ba = Ba[1:PrP.mesh.num_cells(), :]					# Bx.get_shape .toarray()
    # apply algorithm from Preprint
    dof_for_regular_B2 = get_SmartDoFs_CR(PrP.V, PrP.mesh, Ba)

    reg_B2_bloc = Ba[:, dof_for_regular_B2]

    print reg_B2_bloc.toarray()
    print 'Die Determinante ist = %3g' % npla.det(reg_B2_bloc.toarray())
