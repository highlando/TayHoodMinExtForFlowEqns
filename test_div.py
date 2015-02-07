import dolfin
from dolfin import dx, div
import numpy as np

N = 10

mesh = dolfin.UnitSquareMesh(N, N)
# print 'we use Crouzieux-Raviart elements !'
# V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
# Q = dolfin.FunctionSpace(mesh, "DG", 0)
print 'we use Taylor-Hood elements'
V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
Q = dolfin.FunctionSpace(mesh, "CG", 1)

nv = V.dim()

velvec = np.ones((nv, 1))
v = dolfin.Function(V)

v.vector().set_local(velvec)

q = dolfin.TrialFunction(Q)
divv = dolfin.assemble(q*div(v)*dx)
