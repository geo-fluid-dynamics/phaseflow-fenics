""" **refinement.py** contains routines related to local mesh refinement. """
import fenics
import numpy


def adapt_coarse_solution_to_fine_solution(
        scalar,
        coarse_solution, 
        fine_solution,
        element,
        absolute_tolerance = 1.e-2,
        maximum_refinement_cycles = 6,
        circumradius_threshold = 0.01):
    """ Refine the mesh of the coarse solution until the interpolation error tolerance is met. """
    adapted_coarse_mesh = fenics.Mesh(coarse_solution.function_space().mesh())
    
    adapted_coarse_function_space = fenics.FunctionSpace(adapted_coarse_mesh, element)
    
    adapted_coarse_solution = fenics.Function(adapted_coarse_function_space)
    
    adapted_coarse_solution.leaf_node().vector()[:] = coarse_solution.leaf_node().vector()
    
    for refinement_cycle in range(maximum_refinement_cycles):
        
        cell_markers = fenics.MeshFunction(
            "bool", adapted_coarse_mesh.leaf_node(), adapted_coarse_mesh.topology().dim(), False)
        
        for coarse_cell in fenics.cells(adapted_coarse_mesh.leaf_node()):
            
            coarse_value = scalar(adapted_coarse_solution.leaf_node(), coarse_cell.midpoint())
                
            fine_value = scalar(fine_solution.leaf_node(), coarse_cell.midpoint())
            
            if (abs(coarse_value - fine_value) > absolute_tolerance):
            
                cell_markers[coarse_cell] = True
                
        cell_markers = unmark_cells_below_circumradius(
            adapted_coarse_mesh.leaf_node(), cell_markers, circumradius_threshold)
                
        if any(cell_markers):
                
            adapted_coarse_mesh = fenics.refine(adapted_coarse_mesh, cell_markers)
            
            adapted_coarse_function_space = fenics.FunctionSpace(adapted_coarse_mesh, element)
            
            adapted_coarse_solution = fenics.project(
                fine_solution.leaf_node(), 
                adapted_coarse_function_space.leaf_node())
            
        else:
        
            break
    
    return adapted_coarse_solution, adapted_coarse_function_space, adapted_coarse_mesh

    
def unmark_cells_below_circumradius(mesh, cell_markers, circumradius):

    for cell in fenics.cells(mesh):
    
        if not cell_markers[cell]:
            
            continue
            
        if cell.circumradius() <= circumradius:
        
            cell_markers[cell] = False
    
    return cell_markers
