"""This module contains routines for adaptive mesh refinement."""
import fenics
import numpy
import helpers

"""@todo Formalize approach with an error estimator."""

"""fenics requires this exact data structure to evaluate w at a point."""
solution_at_point = numpy.array([1.e32, 1.e32, 1.e32, 1.e32, 1.e32], dtype=numpy.float_) # Oversized for up to 3D

def mark_cells_touching_mushy_region(regularization, mesh, w):
    """Mark cells to refine if they touch the mushy region.
    
    This includes any cell that has at least one vertex both in the
    hot region and the cold region.
    """
    hot = (regularization['theta_s'] + regularization['R_s'] - fenics.dolfin.DOLFIN_EPS)
            
    cold = (regularization['theta_s'] - regularization['R_s'] + fenics.dolfin.DOLFIN_EPS)

    touches_mushy_region = fenics.CellFunction("bool", mesh)

    touches_mushy_region.set_all(False)

    for cell in fenics.cells(mesh):
        
        hot_vertex_count = 0
        
        cold_vertex_count = 0
        
        for vertex in fenics.vertices(cell):
        
            w.eval_cell(solution_at_point, numpy.array([vertex.x(0), vertex.x(1), vertex.x(2)]), cell) # Works for 1/2/3D
            
            theta = solution_at_point[mesh.type().dim() + 1]
            
            if theta > hot:
            
                hot_vertex_count += 1
                
            if theta < cold:
            
                cold_vertex_count += 1

        if (hot_vertex_count < (1 + mesh.type().dim())) and cold_vertex_count < (1 + mesh.type().dim()):
        
            touches_mushy_region[cell] = True
                
    return touches_mushy_region
    
    
def mark_cells_containing_theta_s(regularization, mesh, w):
    """Mark cells to refine if they contain the central fusion temperature.
    
    This includes any cell that has both at least one vertex hotter
    than the fusion temperature and at least one colder.
    """
    theta_s = regularization['theta_s']

    contains_theta_s = fenics.CellFunction("bool", mesh)

    contains_theta_s.set_all(False)

    for cell in fenics.cells(mesh):
        
        hot_vertex_count = 0
        
        cold_vertex_count = 0
        
        for vertex in fenics.vertices(cell):
        
            w.eval_cell(solution_at_point, numpy.array([vertex.x(0), vertex.x(1), vertex.x(2)]), cell) # Works for 1/2/3D
            
            theta = solution_at_point[mesh.type().dim() + 1]
            
            if theta > theta_s:
            
                hot_vertex_count += 1
                
            if theta < theta_s:
            
                cold_vertex_count += 1

        if (hot_vertex_count > 0 and cold_vertex_count > 0):
        
            contains_theta_s[cell] = True
                
    return contains_theta_s
    
    
def refine_pci(regularization, pci_refinement_cycle, mesh, w,
        method='contains_theta_s'):

    if method == 'touches_mushy_region':

        refine_cell = mark_cells_touching_mushy_region(regularization, mesh, w)
    
    elif method == 'contains_theta_s':
    
        refine_cell = mark_cells_containing_theta_s(regularization, mesh, w)

    pci_cell_count = sum(refine_cell)

    assert(pci_cell_count > 0)

    helpers.print_once("PCI Refinement cycle #"+str(pci_refinement_cycle)+
        ": Refining "+str(pci_cell_count)+" cells containing the PCI.")

    mesh = fenics.refine(mesh, refine_cell)
    
    return mesh

    
if __name__=='__main__':

    pass
    