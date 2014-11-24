import numpy as np
from dolfin import Mesh, cells, Cell, facets, Facet
import numpy.linalg as npla


def get_smamin_rearrangement(N, PrP, Mc, Bc, scheme='TH', fullB=None):
    from smamin_utils import col_columns_atend
    from scipy.io import loadmat, savemat
    """ rearrange `B` and `M` for smart minimal extension

    and return the indices of the ext. nodes

    Parameters
    ----------
    scheme : {'TH', 'CR'}
        toggle the scheme

         * 'TH' : Taylor-Hood
         * 'CR' : Crouzeix-Raviart


    Returns
    -------
    MSmeCL : (N,N) sparse matrix
        the rearranged mass matrix (columns and lines swapped)
    BSme : (K, N) sparse matrix
        the rearranged divergence matrix (columns swapped)
    B2Inds : (K, ) array
        indices of the nodes corresponding to the minimal extension
        w.r.t all nodes of the velocity space
    B2BoolInv : (N, ) boolean array
        mask of the ext. nodes w.r.t. the inner nodes in V,
        e.g. `v2 = v[B2BoolInv]`
    B2BI : (K, ) int array
        indices of the ext. nodes w.r.t the inner nodes in V
    """

    if scheme == 'TH':
        print 'solving index 1 -- with TH scheme'
        dname = 'SmeMcBc_N{0}_TH'.format(N)
        get_b2inds_rtn = get_B2_bubbleinds
        args = dict(N=N, V=PrP.V, mesh=PrP.mesh)
    elif scheme == 'CR':
        print 'solving index 1 -- with CR scheme'
        dname = 'SmeMcBc_N{0}_CR'.format(N)
        # pressure-DoF of B_matrix NOT removed yet!
        get_b2inds_rtn = get_B2_CRinds
        args = dict(N=N, V=PrP.V, mesh=PrP.mesh, Q=PrP.Q,
                    B_matrix=Bc, invinds=PrP.invinds)

    try:
        SmDic = loadmat(dname)

    except IOError:
        print 'Computing the B2 indices...'
        # get the indices of the B2-part
        B2Inds = get_b2inds_rtn(**args)
        # the B2 inds wrt to inner nodes
        # this gives a masked array of boolean type
        B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2Inds)
        # this as indices
        B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]
        # Reorder the matrices for smart min ext...
        # ...the columns
        print 'Rearranging the matrices...'
        # Reorder the matrices for smart min ext...
        # ...the columns
        MSmeC = col_columns_atend(Mc, B2BI)
        BSme = col_columns_atend(Bc, B2BI)
        # ...and the lines
        MSmeCL = col_columns_atend(MSmeC.T, B2BI)
        print 'done'

        savemat(dname, {'MSmeCL': MSmeCL,
                        'BSme': BSme,
                        'B2Inds': B2Inds,
                        'B2BoolInv': B2BoolInv,
                        'B2BI': B2BI})

    SmDic = loadmat(dname)

    MSmeCL = SmDic['MSmeCL']
    BSme = SmDic['BSme']
    B2Inds = SmDic['B2Inds']
    B2BoolInv = SmDic['B2BoolInv'] > 0
    B2BoolInv = B2BoolInv.flatten()
    B2BI = SmDic['B2BI']
    only_check_cond = True
    if only_check_cond:
        B2 = BSme[1:, :][:, -B2Inds.size:]
        print 'condition number is ', npla.cond(B2.todense())
        print 'N is ', N
        import matplotlib.pylab as pl
        pl.spy(B2)
        pl.show(block=False)
        # import sys
        # sys.exit('done')

    if fullB is not None:
        fbsme = col_columns_atend(fullB, B2Inds.flatten())
        # fbd = fullB.todense()
        # fb2 = fbd[1:, B2Inds]
        # import matplotlib.pylab as pl
        # pl.spy(fbsme)
        # pl.show(block=False)
        # print 'condition number is ', npla.cond(fb2)
        # print 'N is ', N

    raise Warning('TODO: debug')

    return MSmeCL, BSme, B2Inds, B2BoolInv, B2BI


def getmake_mesh(N):
    """write the mesh for the smart minext tayHood square

    order is I. main grid, II. subgrid = grid of the cluster centers
    and in I and II lexikographical order
    first y-dir, then x-dir """

    try:
        f = open('smegrid%s.xml' % N)
    except IOError:
        print 'Need generate the mesh...'

        # main grid
        h = 1. / (N - 1)
        y, x = np.ogrid[0:N, 0:N]
        y = h * y + 0 * x
        x = h * x + 0 * y
        mgrid = np.hstack((y.reshape(N ** 2, 1), x.reshape(N ** 2, 1)))

        # sub grid
        y, x = np.ogrid[0:N - 1, 0:N - 1]
        y = h * y + 0 * x
        x = h * x + 0 * y
        sgrid = np.hstack(
            (y.reshape(
                (N - 1) ** 2,
                1),
                x.reshape(
                (N - 1) ** 2,
                1)))

        grid = np.vstack((mgrid, sgrid + 0.5 * h))

        f = open('smegrid%s.xml' % N, 'w')
        f.write('<?xml version="1.0"?> \n' +
                '<dolfin xmlns:dolfin="http://www.fenicsproject.org"> \n ' +
                '<mesh celltype="triangle" dim="2"> \n')

        f.write('<vertices size="%s">\n' % (N ** 2 + (N - 1) ** 2))
        for k in range(N ** 2 + (N - 1) ** 2):
            f.write(
                '<vertex index="%s" x="%s" y="%s" />\n' %
                (k, grid[
                    k, 0], grid[
                    k, 1]))

        f.write('</vertices>\n')
        f.write('<cells size="%s">\n' % (4 * (N - 1) ** 2))
        for j in range(N - 1):
            for i in range(N - 1):
                # number of current cluster center
                k = j * (N - 1) + i
                # vertices of the main grid in the cluster
                v0, v1, v2, v3 = j * N + \
                    i, (j + 1) * N + i, (j + 1) * N + i + 1, j * N + i + 1

                f.write(
                    '<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' %
                    (4 * k, v0, N ** 2 + k, v1))
                f.write(
                    '<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' %
                    (4 * k + 1, v1, N ** 2 + k, v2))
                f.write(
                    '<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' %
                    (4 * k + 2, v2, N ** 2 + k, v3))
                f.write(
                    '<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' %
                    (4 * k + 3, v3, N ** 2 + k, v0))

        f.write('</cells>\n')

        f.write('</mesh> \n </dolfin> \n')
        f.close()

        print 'done'

    mesh = Mesh('smegrid%s.xml' % N)

    return mesh


def get_ij_subgrid(k, N):
    """to get i,j numbering of the cluster centers of smaminext"""

    n = N - 1
    if k > n ** 2 - 1 or k < 0:
        raise Exception('%s: No such node on the subgrid!' % k)

    j = np.mod(k, n)
    i = (k - j) / n
    return j, i


def get_B2_bubbleinds(N, V, mesh, Q=None):
    """compute the indices of bubbels that set up

    the invertible B2. This function is specific for the
    mesh generated by smartmintex_tayhoomesh ..."""

    # mesh V must be from
    # mesh = smartminex_tayhoomesh.getmake_mesh(N)
    # V = VectorFunctionSpace(mesh, "CG", 2)

    # 3 bubs * 4 cells * (N-1)**2 cluster
    BubDofs = np.zeros((3 * 4 * (N - 1) ** 2, 4))
    # This will be the array of
    # [x y dofx dofy]

    if Q is None:
        Q = V

    for (i, cell) in enumerate(cells(mesh)):
        # print "Global dofs associated with cell %d: " % i,
        # print Q.dofmap().cell_dofs(i)
        # print "The Dof coordinates:",
        # print Q.dofmap().tabulate_coordinates(cell)
        Cdofs = V.dofmap().cell_dofs(i)
        Coos = V.dofmap().tabulate_coordinates(cell)

        # We sort out the bubble functions - dofs on edge midpoints
        # In every cell the bubbles are numbered 4th-6th (x)
        # and 10th-12th (y-comp)
        CelBubDofs = np.vstack([Cdofs[9:12], Cdofs[3:6]]).T
        CelBubCoos = Coos[3:6]

        BubDofs[i * 3:(i + 1) * 3, :] = np.hstack([CelBubCoos, CelBubDofs])

    # remove duplicate entries
    yDofs = BubDofs[:, -1]
    Aux, IndBubToKeep = np.unique(yDofs, return_index=True)
    BubDofs = BubDofs[IndBubToKeep, :]

    # remove bubbles at cluster boarders
    # x
    XCors = BubDofs[:, 0]
    XCors = np.rint(4 * (N - 1) * XCors)
    IndBorBub = np.in1d(XCors, np.arange(0, 4 * (N - 1) + 1, 4))
    BubDofs = BubDofs[~IndBorBub, :]
    # y
    YCors = BubDofs[:, 1]
    YCors = np.rint(4 * (N - 1) * YCors)
    IndBorBub = np.in1d(YCors, np.arange(0, 4 * (N - 1) + 1, 4))
    BubDofs = BubDofs[~IndBorBub, :]

    # sort by y coordinate
    BubDofs = BubDofs[BubDofs[:, 1].argsort(kind='mergesort')]
    # and by x !!! necessarily by mergesort
    BubDofs = BubDofs[BubDofs[:, 0].argsort(kind='mergesort')]
    # no we have lexicographical order first y then x

    # identify the bubbles of choice
    BD = BubDofs

    VelBubsChoice = np.zeros(0,)
    CI = 2 * (N - 1)  # column increment
    # First column of Cluster
    # First cluster
    ClusCont = np.array([BD[0, 3], BD[1, 2], BD[CI, 2], BD[CI + 1, 3]])
    VelBubsChoice = np.append(VelBubsChoice, ClusCont)

    # loop over the rows
    for iCR in range(1, N - 1):
        ClusCont = np.array([
            BD[2 * iCR + 1, 2],
            BD[2 * iCR + CI, 2],
            BD[2 * iCR + CI + 1, 3]])
        VelBubsChoice = np.append(VelBubsChoice, ClusCont)

    # loop over the columns
    for jCR in range(1, N - 1):
        CC = (2 * jCR) * 2 * (N - 1)  # current column
        # first cluster separate
        ClusCont = np.array([
            BD[CC, 3],
            BD[CC + CI, 2],
            BD[CC + CI + 1, 3]])
        VelBubsChoice = np.append(VelBubsChoice, ClusCont)

        # loop over the rows
        for iCR in range(1, N - 1):
            ClusCont = np.array([
                BD[CC + 2 * iCR + CI, 2],
                BD[CC + 2 * iCR + CI + 1, 3]])
            VelBubsChoice = np.append(VelBubsChoice, ClusCont)

    return VelBubsChoice.astype(int)


# some helpful functions for CR

# returns an array which couples the index of an edge with
# the DoFs of the velocity
#  edgeCRDofArray[edge_index] = [dof_index1, dof_index2]
#  scheinen aber doch sortiert zu sein: edgeCRDofArray[i] = [2i, 2i+1]
def computeEdgeCRDofArray(V, mesh, B=None):
    # dof map, dim_V = 2 * num_E
    num_E = mesh.num_facets()
    dofmap = V.dofmap()
    edgeCRDofArray = np.zeros((num_E, 2))

    # loop over cells and fill array
    for cell in cells(mesh):
        # list of dof-indices for edges of the cell
        dofs = dofmap.cell_dofs(cell.index())
        for i, facet in enumerate(facets(cell)):
            # print 'cell: %3g  ||  i: %3g   ||
            # facet: %3g' % (cell.index(), i, facet.index())
            # corresponding DoFs (2 basisfct per edge)
            edgeCRDofArray[facet.index()] = [dofs[i], dofs[i] + 1]
            # every interior edge visited twice but EGAL!
    return edgeCRDofArray


# finds all adjacent cells
# given a cell and a mesh, it returns the indices of all cells which share
# a common edge with the given cell
def findAdjacentCellInd(cell, mesh):
    adj_cells = []
    D = mesh.topology().dim()
    # Build connectivity between facets and cells
    mesh.init(D - 1, D)
    # loop over edges
    for facet in facets(cell):
        # find all cells with edge facet
        # print facet.entities(D)
        adj_cells = np.append(adj_cells, facet.entities(D))

    # delete doubles and the cell itself
    adj_cells = np.unique(adj_cells)
    adj_cells = adj_cells[adj_cells != cell.index()]

    return adj_cells


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


# returns common edge of two cells (index-version)
# given two cells in terms of the index, it searches for the common edge
# returns the index of the facet
def commonEdgeInd(cell1_ind, cell2_ind, mesh):
    cell1 = Cell(mesh, cell1_ind)
    cell2 = Cell(mesh, cell2_ind)

    return commonEdge(cell1, cell2, mesh).index()


# returns the edges corresponding to V_{2,h} as in the Preprint
#  performs Algorithm 1
#  define mapping iota: cells -> interior edges
def computeSmartMinExtMapping(V, mesh, B=None):
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
        # only adjacent cells which are also in R
        adm_adj_cells = np.intersect1d(adj_cells_last_T, R)

        # it can happen that there is no neoghboring triangle in R
        # then we have to reset last_T
        if len(adm_adj_cells) < 1:
            print ' - Couldnt find adjacent triangles. Have to reset last_T.'
            found_new_triangle = 0
            counter = 0
            while not found_new_triangle:
                test_T = T_minus_R[counter]
                adj_cells_test_T = findAdjacentCellInd(Cell(mesh, test_T),
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
        # update last visited triangle
        last_T = new_T

    return E


def get_B2_CRinds(N=None, V=None, mesh=None, B_matrix=None, invinds=None,
                  Q=None):
    """compute the indices of dofs that set up
    the invertible B2. This function is specific for CR elements."""

    # mesh doesnt matter - can be any mesh
    # V = VectorFunctionSpace(mesh, "CR", 1)
    # koennte man vermutlich auch eleganter ohne die B Matrix machen

    # apply algorithm from Preprint
    edges_V2 = computeSmartMinExtMapping(V, mesh, B=B_matrix)
    # get corresponding degrees of freedom of the CR-scheme
    print 'corresponding DoF for CR'
    edgeCRDofArray = computeEdgeCRDofArray(V, mesh, B=B_matrix)
    DoF_for_V2 = edgeCRDofArray[edges_V2.astype(int)].astype(int)
    DoF_for_V2_x = DoF_for_V2[:, 0]
    # this as indices
    # B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]

    DoF_for_V2_y = DoF_for_V2[:, 1]
    # we still have do decide which of the two basis functions
    # corresponding to the edge we take. Here, we take as default
    # [phi_E; 0] if not div[phi_E; 0] = 0
    # - This we check via the norm of the column in B
    dof_for_regular_B2 = DoF_for_V2_x

    lut = dict(zip(invinds.tolist(), range(invinds.size)))

    for i in np.arange(len(edges_V2)):
        # take x-DoF and test whether its a zero-column
        xdof = lut[DoF_for_V2_x[i]]
        ydof = lut[DoF_for_V2_y[i]]
        colx = B_matrix[:, xdof]
        coly = B_matrix[:, ydof]
        if abs(coly[i+1, 0]) > abs(colx[i+1, 0]):
            dof_for_regular_B2[i] = DoF_for_V2_y[i]

        # # Problem, dass erster Eintrag noch da?? vmtl nein
        #
        # if npla.norm(col.toarray(), np.inf) < 1e-14:
        #     # norm to small --> seems to be a zero-column,
        #     # i.e., divergence vanishes
        #     dof_for_regular_B2[i] = DoF_for_V2_y[i]

    # raise Warning('TODO: debug')
    return dof_for_regular_B2
