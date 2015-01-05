import dolfin
import dolfin_navier_scipy.problem_setups as dnsps

N = 10
mesh = dolfin.UnitSquareMesh(N, N)

edgepoint = dolfin.Point(1., 1.)
aaa = mesh.bounding_box_tree().compute_first_entity_collision(edgepoint)

femp, stokesmatsc, rhsd_vfrc, \
    rhsd_stbc, data_prfx, ddir, proutdir \
    = dnsps.get_sysmats(problem='cylinderwake', N=2, Re=2, scheme='CR')

mesh = femp['mesh']
edgepoint = dolfin.Point(2.2, .41)
aaac = mesh.bounding_box_tree().compute_first_entity_collision(edgepoint)
dolfin.plot(mesh)
dolfin.interactive(True)
