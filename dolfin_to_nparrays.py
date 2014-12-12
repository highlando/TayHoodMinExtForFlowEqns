from dolfin import grad, inner, dx, div, assemble
import dolfin
import numpy as np
import scipy.sparse as sps

dolfin.parameters.linear_algebra_backend = "uBLAS"


def get_sysNSmats(V, Q, nu=1.):  # , velbcs ):
    """ Assembles the system matrices for Stokes equation

    in mixed FEM formulation, namely

            [ A  B' ] as [ Aa   Grada ] : W -> W'
            [ B  0  ]    [ Diva   0   ]

    for a given trial and test space W = V * Q and boundary conds.

    Plus the velocity mass matrix M.
    """

    u = dolfin.TrialFunction(V)
    p = dolfin.TrialFunction(Q)
    v = dolfin.TestFunction(V)
    q = dolfin.TestFunction(Q)

    ma = inner(u, v) * dx
    mp = inner(p, q) * dx
    # we multiply by nu below for the case that nu=0
    aa = inner(grad(u), grad(v)) * dx
    grada = div(v) * p * dx
    diva = q * div(u) * dx

    # Assemble system
    M = assemble(ma)
    A = assemble(aa)
    Grad = assemble(grada)
    Div = assemble(diva)
    MP = assemble(mp)

    # Convert DOLFIN representation to numpy arrays
    rows, cols, values = M.data()
    Ma = sps.csr_matrix((values, cols, rows))
    Ma.eliminate_zeros()

    rows, cols, values = MP.data()
    MPa = sps.csr_matrix((values, cols, rows))
    MPa.eliminate_zeros()

    rows, cols, values = A.data()
    Aa = nu*sps.csr_matrix((values, cols, rows))
    Aa.eliminate_zeros()

    rows, cols, values = Grad.data()
    BTa = sps.csr_matrix((values, cols, rows))
    BTa.eliminate_zeros()

    rows, cols, values = Div.data()
    Ba = sps.csr_matrix((values, cols, rows))
    Ba.eliminate_zeros()

    return Ma, Aa, BTa, Ba, MPa


def setget_rhs(V, Q, fv, fp, velbcs=None, t=None):

    v = dolfin.TestFunction(V)
    q = dolfin.TestFunction(Q)

    fv = inner(fv, v) * dx
    fp = inner(fp, q) * dx

    fv = assemble(fv)
    fp = assemble(fp)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    fp = fp.array()
    fp = fp.reshape(len(fp), 1)

    return fv, fp


def get_curfv(V, fv, invinds, tcur):
    """get the fv at innernotes at t=tcur

    """

    v = dolfin.TestFunction(V)

    fv.t = tcur

    fv = inner(fv, v) * dx

    fv = assemble(fv)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    return fv[invinds, :]


def get_convvec(u0, V):
    """return the convection vector e.g. for explicit schemes"""

    v = dolfin.TestFunction(V)
    ConvForm = inner(grad(u0) * u0, v) * dx

    ConvForm = assemble(ConvForm)
    ConvVec = ConvForm.array()
    ConvVec = ConvVec.reshape(len(ConvVec), 1)

    return ConvVec


def condense_sysmatsbybcs(Ma, Aa, BTa, Ba, fv, fp, velbcs):
    """resolve the Dirichlet BCs and condense the sysmats to the inner nodes"""

    auxu = np.zeros((len(fv), 1))					# Spaltenvektor statt Zeilenvektor
    bcinds = []
    for bc in velbcs:
        bcdict = bc.get_boundary_values()
        # save bc values at correct pos
        auxu[bcdict.keys(), 0] = bcdict.values()
        bcinds.extend(bcdict.keys())

    # putting the bcs into the right hand sides
    fvbc = - Aa * auxu    # '*' is np.dot for csr matrices
    fpbc = - Ba * auxu

    # indices of the innernodes
    innerinds = np.setdiff1d(range(len(fv)), bcinds).astype(np.int32)

    # extract the inner nodes equation coefficients
    Mc = Ma[
        innerinds, :][
        :, innerinds]				# NOT equal to Ma[innerinds,innerinds] !!
    Ac = Aa[innerinds, :][:, innerinds]
    fvbc = fvbc[innerinds, :]
    Bc = Ba[:, innerinds]
    BTc = BTa[innerinds, :]

    bcvals = auxu[bcinds]

    return Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, innerinds
