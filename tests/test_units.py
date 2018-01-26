""" This module runs the unit test suite. """
from __future__ import print_function
from builtins import str
from .context import phaseflow
import fenics


def test_1d_output_unit():

    mesh = fenics.UnitIntervalMesh(5)
    
    mixed_element = phaseflow.pure.make_mixed_element(mesh.ufl_cell())
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    state = phaseflow.core.State(function_space)
    
    with phaseflow.core.SolutionFile("output/test_1D_output/solution.xdmf") as solution_file:
     
        state.write_solution_to_xdmf(solution_file)
    
        
def test_1d_velocity_unit():

    mesh = fenics.UnitIntervalMesh(5)

    V = fenics.VectorFunctionSpace(mesh, "P", 1)

    u = fenics.Function(V)

    bc = fenics.DirichletBC(V, [10.0], "x[0] < 0.5")

    print(bc.get_boundary_values())
    
    
def test_xdmf_unit():

    solution_file = fenics.XDMFFile("test.xdmf")


'''This test seems to fail with fenics-2016.2.0. 
I vaguely recall seeing an issue on their Bitbucket which mentions having
not always used the proper context manager style with some of their file classes.'''
def test_xdmf_context_unit():

    with fenics.XDMFFile("test.xdmf") as solution_file:

        return
        

if __name__=="__main__":
    
    test_1d_output_unit()
    
    test_1d_velocity_unit()
    
    test_xdmf_unit()

    test_xdmf_context_unit()
