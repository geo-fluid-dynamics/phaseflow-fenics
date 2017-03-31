# Local refinement example from https://fenicsproject.org/qa/9760/what-is-the-best-approach-to-mesh-refinement-on-boundary

from dolfin import *

initial_grid_M = 10

mesh = UnitSquareMesh(initial_grid_M, initial_grid_M, "crossed")

# Selecting edges to refine
class Wall(SubDomain):
    
    def inside(self, x, on_boundary):
    
        return on_boundary and (near(x[0], 0.) or near(x[0], 1.) or near(x[1], 0.) or near(x[1], 1.))

        
Wall = Wall()

# Number of refinements
refinement_cycles = 3

for i in range(refinement_cycles):
    
    edge_markers = EdgeFunction("bool", mesh)
    
    Wall.mark(edge_markers, True)

    adapt(mesh, edge_markers)
    
    mesh = mesh.child()

    
Q = FunctionSpace(mesh, 'P', 1)

q = Function(Q)

file = File('output/values_on_mesh.pvd')

file << (q) 
