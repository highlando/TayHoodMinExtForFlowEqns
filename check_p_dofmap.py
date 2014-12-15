# import numpy as np
import dolfin
from dolfin import cells

N = 3
mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.VectorFunctionSpace(mesh, 'CR', 1)

Q = dolfin.FunctionSpace(mesh, 'DG', 0)

# p = dolfin.Function(Q)
# p.vector().set_local(np.linspace(0, 1, Q.dim()))
# dolfin.plot(p)
# dolfin.interactive(True)

for (i, cell) in enumerate(cells(mesh)):
    # print "Global dofs associated with cell %d: " % i,
    # print Q.dofmap().cell_dofs(i)
    # print "The Dof coordinates:",
    # print Q.dofmap().tabulate_coordinates(cell)
    print 'V dof', V.dofmap().cell_dofs(i)
    print 'V coor', V.dofmap().tabulate_coordinates(cell)

# for (i, cell) in enumerate(cells(mesh)):
#     print '\ndof\n', Q.dofmap().cell_dofs(i)
#     print 'coor\n', Q.dofmap().tabulate_coordinates(cell)
