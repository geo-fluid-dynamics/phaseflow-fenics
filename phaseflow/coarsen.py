""" This module implements a coarsening algorithm for Phaseflow based on cell-wise interpolation error.

For simplicity, for now we'll consider only one scalar component of the solution.
"""
import phaseflow
import fenics
import numpy


def adapt_solution_to_solution_on_other_mesh(coarse_solution, 
        fine_solution,
        absolute_tolerance = 1.e-2,
        maximum_refinement_cycles = 6,
        scalar_solution_component_index = 3):
    """ Refine the mesh of the coarse solution until the interpolation error tolerance is met. """
    for refinement_cycle in range(maximum_refinement_cycles):
    
        coarse_mesh = coarse_solution.function_space().mesh()
    
        exceeds_tolerance = fenics.CellFunction("bool", coarse_mesh.leaf_node())

        exceeds_tolerance.set_all(False)
    
        for coarse_cell in fenics.cells(coarse_mesh):
            
            coarse_value = coarse_solution.leaf_node()(coarse_cell.midpoint())\
                [scalar_solution_component_index]
                
            fine_value = fine_solution.leaf_node()(coarse_cell.midpoint())\
                [scalar_solution_component_index]
            
            if (abs(coarse_value - fine_value) > absolute_tolerance):
            
                exceeds_tolerance[coarse_cell] = True
                
        if any(exceeds_tolerance):
                
            coarse_mesh = fenics.refine(coarse_mesh, exceeds_tolerance)
            
            coarse_solution = fenics.project(fine_solution, coarse_solution.leaf_node().function_space())
            
        else:
        
            break
    
    return coarse_solution  # This should reference the refined coarse mesh.
    