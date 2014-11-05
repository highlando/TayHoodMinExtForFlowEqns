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


# problem = LinearVariationalProblem(a, L, u, bc)
# solver = LinearVariationalSolver(problem)
# solver.parameters["linear_solver"] = "cg"
# solver.parameters["preconditioner"] = "ilu"
# cg_prm = solver.parameters["krylov_solver"]
# cg_prm["absolute_tolerance"] = 1E-7
# cg_prm["relative_tolerance"] = 1E-4
# cg_prm["maximum_iterations"] = 150
# solver.solve()


# Ausgabe der Koordinaten
# coord = mesh.coordinates()
# if mesh.num_vertices() == len(u_array):
#   for i in range(mesh.num_vertices()):
# print 'Koordinate (%3g, %3g) mit Wert %6g' % (coord[i][0], coord[i][1],
# u_array[i])

# Manipulate Data
# max_val = u_array.max()
# u_array /= max_val
# u.vector()[:] = u_array

# Berechnung von Integralen/Energien
# energy = 0.5*inner(grad(u), grad(u))*dx
# E = assemble(energy)

# bc.apply(A, b)						# (unsymmetric) modifications due to bc

# FEniCS for nonlinear problems (Newton iteration, page 44)
# u = TrialFunction(V)
# v = TestFunction(V)
# F = innner(q(u)*nabla_grad(u), grad_nabla(v))*dx
# u_ = Function(V)					# Loesung im aktuellen Newton schritt
# F = action(F, u_)					# equals F(u=u_ ;v)
# du = TrialFunction(V)
# J = inner(q(u_)*nabla_grad(du), ...			# Jacobian
# J = derivative(F, u_, du)				# Dateaux derivative in direction of du
# problem = NonlinearVariationalProblem(F, u_, bcs, J)


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


# some common functions

# FERTIG!
# returns an array which couples the index of an edge with
# the DoFs of the velocity
#  edgeCRDofArray[edge_index] = [dof_index1, dof_index2]
#  scheinen aber doch sortiert zu sein: edgeCRDofArray[i] = [2i, 2i+1]
def computeEdgeCRDofArray(V, mesh):
    # dof map
    num_E = mesh.num_facets()	 				# dim_V = 2 * num_E
    dofmap = V.dofmap()
    edgeCRDofArray = np.zeros((num_E, 2))

    # loop over cells and fill array
    for cell in cells(mesh):
        # list of dof-indices for edges of the cell
        dofs = dofmap.cell_dofs(cell.index())
        for i, facet in enumerate(facets(cell)):
            # print 'cell: %3g  ||  i: %3g   || facet: %3g' % (cell.index(), i,
            # facet.index())
            edgeCRDofArray[
                facet.index()] = [
                dofs[i],
                dofs[i] +
                1]  # corresponding DoFs (2 basisfct per edge)
            # every interior edge visited twice but EGAL!
    return edgeCRDofArray


# FERTIG!
# finds all adjacent cells
# given a cell and a mesh, it returns the indices of all cells which share
# a common edge with the given cell
def findAdjacentCellInd(cell, mesh):
    adj_cells = []
    D = mesh.topology().dim()
    mesh.init(D - 1, D) 						# Build connectivity between facets and cells
    # loop over edges
    for facet in facets(cell):
        # find all cells with edge facet
        # print facet.entities(D)
        adj_cells = np.append(adj_cells, facet.entities(D))

    # delete doubles and the cell itself
    adj_cells = np.unique(adj_cells)
    # adj_cells = np.delete(adj_cells, dieser_index)          #
    # np.delete(array, position_to_delete)
    adj_cells = adj_cells[adj_cells != cell.index()]  		# so klappt es !!

    return adj_cells


# FERTIG!
# returns common edge of two cells
# given two Cell objects, it searches for the common edge
# returns the Facet object
def commonEdge(cell1, cell2, mesh):
    facets1 = facets(cell1)
    facets2 = facets(cell2)

    # find common index
    ind_fac1 = []
    for facet in facets1:
        ind_fac1 = np.append(ind_fac1, facet.index())
    ind_fac2 = []
    for facet in facets2:
        ind_fac2 = np.append(ind_fac2, facet.index())

    # intersect gives index
    index = np.intersect1d(ind_fac1, ind_fac2)

    return Facet(mesh, int(index[0]))

# FERTIG!
# returns common edge of two cells (index-version)
# given two cells in terms of the index, it searches for the common edge
# returns the index of the facet


def commonEdgeInd(cell1_ind, cell2_ind, mesh):
    cell1 = Cell(mesh, cell1_ind)
    cell2 = Cell(mesh, cell2_ind)

    return commonEdge(cell1, cell2, mesh).index()


# FERTIG!
# returns the edges corresponding to V_{2,h} as in the Preprint
#  performs Algorithm 1
#  define mapping iota: cells -> interior edges
def computeSmartMinExtMapping(V, mesh):
    nr_cells = mesh.num_cells()
    # T_0 = triangle with sell-index 0
    # list of remaining triangles
    R = np.arange(nr_cells)
    R = R[1:mesh.num_cells()]
    # list of already visited triangles and selected edges
    T_minus_R = [0]
    E = []
    # indox of 'last' triangle
    last_T = 0

    # loop until to triangles are left
    print 'Enter while loop'
    while (len(R) > 0):
        # find adjacent triangle of last_T
        adj_cells_last_T = findAdjacentCellInd(Cell(mesh, last_T), mesh)
        adm_adj_cells = np.intersect1d(
            adj_cells_last_T,
            R)		# only adjacent cells which are also in R

        # it can happen that there is no neoghboring triangle in R
        # then we have to reset last_T
        if len(adm_adj_cells) < 1:
            print ' - Couldnt find adjacent triangles. Have to reset last_T.'
            found_new_triangle = 0
            counter = 0
            while not found_new_triangle:
                test_T = T_minus_R[counter]
                adj_cells_test_T = findAdjacentCellInd(
                    Cell(
                        mesh,
                        test_T),
                    mesh)
                # print np.intersect1d(adj_cells_test_T, R)
                if len(np.intersect1d(adj_cells_test_T, R)) > 0:
                    print ' - - YES! I found a new triangle.'
                    found_new_triangle = 1
                    last_T = test_T
                    adm_adj_cells = np.intersect1d(adj_cells_test_T, R)
                counter = counter + 1

        # if there exists at least one admissible neighbor: get common edge
        new_T = int(adm_adj_cells[0])
        print 'old Tri %3g and new found Tri %3g' % (last_T, new_T)
        R = R[R != new_T]
        T_minus_R = np.append(T_minus_R, new_T)
        comm_edge = commonEdgeInd(last_T, new_T, mesh)
        # update range(iota), i.e., list of edges
        E = np.append(E, comm_edge)
        last_T = new_T						# update last visited triangle

    return E


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
