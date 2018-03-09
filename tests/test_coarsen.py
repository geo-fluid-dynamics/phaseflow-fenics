""" This module tests Phaseflow's coarsening routine. """
from .context import phaseflow
import fenics
import tempfile


def test_coarsen_1d__ci__():
    """ This tests mesh coarsening in 1D. """
    sim = phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation()
    
    sim.run()
    
    newsim = phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation()
    
    newsim.update_coarse_mesh()
    
    newsim.update_element()
    
    newsim.update_function_space()
    
    newsim.update_states()
    
    phaseflow.coarsen.adapt_solution_to_solution_on_other_mesh(newsim.state.solution, 
        sim.state.solution,
        absolute_tolerance = sim.absolute_tolerance,
        maximum_refinement_cycles = sim.initial_hot_boundary_refinement_cycles,
        scalar_solution_component_index = 2)
    
    newsim.verify()
    
    
def test_coarsen_2d():
    """ This tests mesh coarsening in 2D. """
    sim = phaseflow.octadecane_benchmarks.CCMOctadecanePCMRegressionSimulation()
    
    sim.run()
    
    newsim = phaseflow.octadecane_benchmarks.CCMOctadecanePCMRegressionSimulation()
    
    newsim.update_coarse_mesh()
    
    newsim.update_element()
    
    newsim.update_function_space()
    
    newsim.update_states()
    
    newsim.state.solution = phaseflow.coarsen.adapt_solution_to_solution_on_other_mesh(
        newsim.state.solution, 
        sim.state.solution,
        absolute_tolerance = 1.e-2,
        maximum_refinement_cycles = 5,
        scalar_solution_component_index = 3)
    
    newsim.output_dir += "coarsened/"
    
    with phaseflow.helpers.SolutionFile(newsim.output_dir + "/solution.xdmf") as solution_file:
        
        newsim.state.write_solution(solution_file)
    
    newsim.verify()
    