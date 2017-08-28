import fenics
import numpy

import helpers


solution_at_point = numpy.array([1.e32, 1.e32, 1.e32, 1.e32, 1.e32], dtype=numpy.float_) # Oversized for up to 3D

def mark_pci_cells(regularization, mesh, w):

    hot = (regularization['theta_s'] + regularization['R_s'] - fenics.dolfin.DOLFIN_EPS)
            
    cold = (regularization['theta_s'] - regularization['R_s'] + fenics.dolfin.DOLFIN_EPS)

    touches_pci = fenics.CellFunction("bool", mesh)

    touches_pci.set_all(False)

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
        
            touches_pci[cell] = True
                
    return touches_pci
    
    
def refine_pci(regularization, pci_refinement_cycle, mesh, w):

    contains_pci = mark_pci_cells(regularization, mesh, w)

    pci_cell_count = sum(contains_pci)

    assert(pci_cell_count > 0)

    helpers.print_once("PCI Refinement cycle #"+str(pci_refinement_cycle)+
        ": Refining "+str(pci_cell_count)+" cells containing the PCI.")

    mesh = fenics.refine(mesh, contains_pci)
    
    return mesh
