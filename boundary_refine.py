# Local refinement example from https://fenicsproject.org/qa/9760/what-is-the-best-approach-to-mesh-refinement-on-boundary

from dolfin import *

mesh = UnitSquareMesh(20, 20, "crossed")

# Break point
p   = Point(0.0, 0.5)
tol = 0.05

# Selecting edges to refine
class Border(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], p.x(), tol) and near(x[1], p.y(), tol) and on_boundary

Border = Border()

# Number of refinements
nor = 3

for i in range(nor):
    edge_markers = EdgeFunction("bool", mesh)
    Border.mark(edge_markers, True)

    adapt(mesh, edge_markers)
    mesh = mesh.child()

    
Q = FunctionSpace(mesh, 'P', 1)

q = Function(Q)

file = File('output/values_on_mesh.pvd')

file << (q) 
