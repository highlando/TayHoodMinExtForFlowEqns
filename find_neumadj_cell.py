import dolfin

N = 10
mesh = dolfin.UnitSquareMesh(N, N)

edgepoint = dolfin.Point(1., 1.)

aaa = mesh.bounding_box_tree().compute_first_entity_collision(edgepoint)

dolfin.plot(mesh)

dolfin.interactive(True)
