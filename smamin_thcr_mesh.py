import numpy as np
from dolfin import Mesh, cells, Cell, facets, Facet
import numpy.linalg as npla
import scipy.sparse as sps


def get_smamin_rearrangement(N, PrP, V=None, Q=None, invinds=None, nu=None,
                             Pdof=None, M=None, A=None, B=None, mesh=None,
                             addnedgeat=None,
                             scheme='TH', fullB=None, crinicell=None):
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

    crinicell : int, optional
        the starting cell for the 'CR' scheme, defaults to `0`
    addnedge : int, optional
        whether to add a Neumann edge in the CR scheme, defaults to `None`

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
    Q = PrP.Q if Q is None else Q
    V = PrP.V if V is None else V
    invinds = PrP.invinds if invinds is None else invinds
    nu = PrP.nu if nu is None else nu
    mesh = PrP.mesh if mesh is None else mesh
    Pdof = PrP.Pdof if Pdof is None and PrP is not None else Pdof

    if scheme == 'TH':
        print 'solving index 1 -- with TH scheme'
        dname = 'mats/SmeMcBc_N{0}nu{1}_TH'.format(N, nu)
        get_b2inds_rtn = get_B2_bubbleinds
        args = dict(N=N, V=V, mesh=mesh)
    elif scheme == 'CR':
        print 'solving index 1 -- with CR scheme'
        dname = 'mats/SmeMcBc_N{0}nu{1}_CR'.format(N, nu)
        # pressure-DoF of B_matrix NOT removed yet!
        get_b2inds_rtn = get_B2_CRinds
        args = dict(N=N, V=V, mesh=mesh, Q=Q, inicell=crinicell,
                    B_matrix=B, invinds=invinds)

    try:
        SmDic = loadmat(dname)
        pdoflist = loadmat(dname+'pdoflist')  # TODO enable saving again

    except IOError:
        print 'Computing the B2 indices...'
        # get the indices of the B2-part
        B2Inds, pdoflist = get_b2inds_rtn(**args)
        if addnedgeat is not None:
            # TODO: hard coded filtering of the needed V bas func
            # list of columns that have a nnz at cell #addnedge
            potcols = fullB[addnedgeat, :].indices
            for col in potcols:
                # TODO here we need B
                if fullB[:, col].nnz == 1:
                    coltoadd = col
                    break
            B2Inds = np.r_[coltoadd, B2Inds]
            # end TODO

        # the B2 inds wrt to inner nodes
        # this gives a masked array of boolean type
        B2BoolInv = np.in1d(np.arange(V.dim())[invinds], B2Inds)
        # this as indices
        B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]
        # Reorder the matrices for smart min ext...
        # ...the columns
        print 'Rearranging the matrices...'
        # Reorder the matrices for smart min ext...
        # ...the columns
        MSmeC = col_columns_atend(M, B2BI)
        BSme = col_columns_atend(B, B2BI)
        # ...and the lines
        MSmeCL = col_columns_atend(MSmeC.T, B2BI)
        if A is not None:
            ASmeC = col_columns_atend(A, B2BI)
            ASmeCL = (col_columns_atend(ASmeC.T, B2BI)).T

        print 'done'

        savemat(dname, {'MSmeCL': MSmeCL,
                        'ASmeCL': ASmeCL,
                        'BSme': BSme,
                        'B2Inds': B2Inds,
                        'B2BoolInv': B2BoolInv,
                        'B2BI': B2BI})
        if scheme == 'CR':
            savemat(dname+'pdoflist', {'pdoflist': pdoflist})

    SmDic = loadmat(dname)

    MSmeCL = SmDic['MSmeCL']
    ASmeCL = SmDic['ASmeCL']
    BSme = SmDic['BSme']
    B2Inds = SmDic['B2Inds']
    B2BoolInv = SmDic['B2BoolInv'] > 0
    B2BoolInv = B2BoolInv.flatten()
    B2BI = SmDic['B2BI']
    if scheme == 'CR':
        pdoflist = loadmat(dname+'pdoflist')['pdoflist']
    else:
        pdoflist = None
    only_check_cond = False
    if only_check_cond:
        print 'Scheme is ', scheme
        import matplotlib.pylab as pl
        if Pdof is None:
            B2 = BSme[:, :][:, -B2Inds.size:]
            B2res = fullB[pdoflist.flatten(), :][:, B2Inds.flatten()]
            print 'condition number is ', npla.cond(B2res.todense())
            # B2res = BSme[pdoflist.flatten(), :][:, -B2Inds.size:]
            pl.figure(2)
            pl.spy(B2res)  # [:100, :][:, :100])
        elif Pdof == 0:
            B2 = BSme[1:, :][:, -B2Inds.size:]
            print 'condition number is ', npla.cond(B2.todense())
        else:
            raise NotImplementedError()
        print 'N is ', N
        print 'B2 shape is ', B2.shape
        pl.figure(1)
        pl.spy(B2)
        pl.show(block=False)
        import sys
        sys.exit('done')

    if fullB is not None and only_check_cond:
        fbsme = col_columns_atend(fullB, B2Inds.flatten())
        import matplotlib.pylab as pl
        pl.figure(2)
        pl.spy(fbsme)
        pl.show(block=False)
        fbsmec = fbsme[0:, :][:, -B2Inds.size:]
        pl.figure(3)
        pl.spy(fbsmec)
        pl.show(block=False)
        if pdoflist is not None:
            linelist = []
            for pdof in pdoflist.flatten().tolist()[1:]:
                linelist.append(fbsmec[pdof, :])
            fbsmecr = sps.vstack(linelist)
        pl.figure(4)
        pl.spy(fbsmecr)
        pl.show(block=False)

        print 'condition number is ', npla.cond(fbsmecr.T.todense())
        print 'N is ', N

    if A is None:
        return MSmeCL, BSme, B2Inds, B2BoolInv, B2BI
    else:
        return MSmeCL, ASmeCL, BSme, B2Inds, B2BoolInv, B2BI


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
    mesh generated by smamin_thcr_mesh ..."""

    # mesh V must be from
    # mesh = smamin_thcr_mesh.getmake_mesh(N)
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

    return VelBubsChoice.astype(int), None


# some helpful functions for CR

# returns an array which couples the index of an edge with
# the DoFs of the velocity
#  edgeCRDofArray[edge_index] = [dof_index1, dof_index2]
#  scheinen aber doch sortiert zu sein: edgeCRDofArray[i] = [2i, 2i+1]
# nicht in fenics v1.2 -- chngd to edgeCRDofArray[i] = [dof[i], dof[i+3]]
def computeEdgeCRDofArray(V, mesh, B=None):
    # dof map, dim_V = 2 * num_E
    num_E = mesh.num_facets()
    dofmap = V.dofmap()
    edgeCRDofArray = np.zeros((num_E, 2))

    # loop over cells and fill array
    for k, cell in enumerate(cells(mesh)):
        # list of dof-indices for edges of the cell
        dofs = dofmap.cell_dofs(cell.index())
        for i, facet in enumerate(facets(cell)):
            # print 'cell: %3g  ||  i: %3g   || facet: %3g || dof[i]: %3g' \
            #     % (cell.index(), i, facet.index(), dofs[i])
            # corresponding DoFs (2 basisfct per edge)
            edgeCRDofArray[facet.index()] = [dofs[i], dofs[i+3]]
            # edgeCRDofArray[facet.index()] = [dofs[i], dofs[i]+1]
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
def computeSmartMinExtMapping(V, mesh, B=None, Tzero=None):
    nr_cells = mesh.num_cells()
    # T_0 = starting triangle with given cell-index
    # list of already visited triangles and selected edges
    if Tzero is None:
        Tzero = 0
    T_minus_R = [Tzero]
    # list of remaining triangles
    R = np.arange(nr_cells)
    if Tzero == 0:
        R = R[1:mesh.num_cells()]
    else:
        R = R[R != Tzero]  # TODO: do this efficiently since R is sortd
    E = []
    # index of 'last' triangle
    last_T = Tzero

    # loop until to triangles are left
    print 'Enter while loop'
    while (len(R) > 0):
        # find adjacent triangle of last_T
        adj_cells_last_T = findAdjacentCellInd(Cell(mesh, last_T), mesh)
        # only adjacent cells which are also in R
        adm_adj_cells = np.intersect1d(adj_cells_last_T, R)

        # it can happen that there is no neighboring triangle in R
        # then we have to reset last_T
        if len(adm_adj_cells) < 1:
            print ' - Couldnt find adjacent triangles. Have to reset last_T.'
            found_new_triangle = False
            counter = 0
            while not found_new_triangle:
                test_T = T_minus_R[counter]
                adj_cells_test_T = findAdjacentCellInd(Cell(mesh, test_T),
                                                       mesh)
                # print np.intersect1d(adj_cells_test_T, R)
                if len(np.intersect1d(adj_cells_test_T, R)) > 0:
                    print ' - - YES! I found a new triangle.'
                    found_new_triangle = True
                    last_T = test_T
                    adm_adj_cells = np.intersect1d(adj_cells_test_T, R)
                counter = counter + 1

        # if there exists at least one admissible neighbor: get common edge
        new_T = int(adm_adj_cells[0])
        print 'old Tri %3g and new found Tri %3g' % (last_T, new_T)
        R = R[R != new_T]  # TODO: this can be done more efficient R is sorted
        T_minus_R = np.append(T_minus_R, new_T)
        comm_edge = commonEdgeInd(last_T, new_T, mesh)
        # update range(iota), i.e., list of edges
        E = np.append(E, comm_edge)
        # update last visited triangle
        last_T = new_T

    return E, T_minus_R


def get_B2_CRinds(N=None, V=None, mesh=None, B_matrix=None, invinds=None,
                  Q=None, inicell=0):
    """compute the indices of dofs that set up
    the invertible B2. This function is specific for CR elements."""

    # mesh doesnt matter - can be any mesh
    # V = VectorFunctionSpace(mesh, "CR", 1)
    # koennte man vermutlich auch eleganter ohne die B Matrix machen

    # apply algorithm from Preprint
    edges_V2, pdoflist = computeSmartMinExtMapping(V, mesh, B=B_matrix,
                                                   Tzero=inicell)
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
        if np.linalg.norm(coly.todense()) > np.linalg.norm(colx.todense()):
            dof_for_regular_B2[i] = DoF_for_V2_y[i]

    return dof_for_regular_B2, pdoflist


def get_cellid_nexttopoint(mesh, point):
    return mesh.bounding_box_tree().compute_first_entity_collision(point)
