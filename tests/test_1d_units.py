from __future__ import print_function
from builtins import str
from .context import phaseflow

import fenics
        
        
def test_1d_output():

    mesh = fenics.UnitIntervalMesh(10)
    
    mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    boundaries = "near(x[0],  0.) | near(x[0],  1.)"
    
    phaseflow.run(solution = fenics.Function(function_space),
        initial_values = fenics.Function(function_space),
        boundary_conditions = None,
        output_dir = "output/test_1D_output/",
        gravity = [0.],
        end_time = 0.)

        
def test_1d_velocity():

    mesh = fenics.UnitIntervalMesh(fenics.dolfin.mpi_comm_world(), 5)

    V = fenics.VectorFunctionSpace(mesh, "P", 1)

    u = fenics.Function(V)

    bc = fenics.DirichletBC(V, [10.0], "x[0] < 0.5")

    print(bc.get_boundary_values())
    

if __name__=="__main__":
    
    test_1d_output()
    
    test_1d_velocity()
