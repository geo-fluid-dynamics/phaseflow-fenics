""" This module runs the unit test suite. """
from .context import phaseflow
import fenics
import tempfile


def test_1d_output_unit__ci__():

    sim = phaseflow.phasechange_simulation.PhaseChangeSimulation()
    
    sim.mesh = fenics.UnitIntervalMesh(5)
    
    sim.setup_element()
    
    function_space = fenics.FunctionSpace(sim.mesh, sim.element)
    
    state = phaseflow.state.State(function_space, sim.element)
    
    with phaseflow.helpers.SolutionFile(tempfile.mkdtemp() + "/output/test_1D_output/solution.xdmf") \
            as solution_file:
     
        state.write_solution(solution_file)
    
        
def test_1d_velocity_unit__ci__():

    mesh = fenics.UnitIntervalMesh(5)

    V = fenics.VectorFunctionSpace(mesh, "P", 1)

    u = fenics.Function(V)

    bc = fenics.DirichletBC(V, [10.0], "x[0] < 0.5")

    print(bc.get_boundary_values())
    
    
def test_xdmf_unit__ci__():

    solution_file = fenics.XDMFFile("test.xdmf")


'''This test seems to fail with fenics-2016.2.0. 
I vaguely recall seeing an issue on their Bitbucket which mentions having
not always used the proper context manager style with some of their file classes.'''
def test_xdmf_context_unit__ci__():

    with fenics.XDMFFile("test.xdmf") as solution_file:

        return
