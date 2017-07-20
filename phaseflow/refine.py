import fenics
import numpy


solution_at_point = numpy.array([1.e32, 1.e32, 1.e32, 1.e32, 1.e32], dtype=numpy.float_)

def mark_pci_cells(regularization, mesh, w):

    contains_pci = fenics.CellFunction("bool", mesh)

    contains_pci.set_all(False)

    for cell in fenics.cells(mesh):
        
        hot_vertex_count = 0
        
        cold_vertex_count = 0
        
        for vertex in fenics.vertices(cell):
        
            w.eval_cell(solution_at_point, numpy.array([vertex.x(0), vertex.x(1), vertex.x(2)]), cell)
            
            if mesh.type().dim() is 1:
            
                theta = solution_at_point[2]
                
            elif mesh.type().dim() is 2:
            
                theta = solution_at_point[3]
                
            hot = (regularization['theta_s'] + 2*regularization['R_s'] - fenics.dolfin.DOLFIN_EPS)
            
            cold = (regularization['theta_s'] - 2*regularization['R_s'] + fenics.dolfin.DOLFIN_EPS)
            
            if theta > hot:
            
                hot_vertex_count += 1
                
            if theta < cold:
            
                cold_vertex_count += 1

        if mesh.type().dim() is 1:
        
            if (hot_vertex_count is 1) or (cold_vertex_count is 1):
        
                contains_pci[cell] = True
                
        elif mesh.type().dim() is 2:
        
            if (0 < hot_vertex_count < 3) or (0 < cold_vertex_count < 3):
            
                contains_pci[cell] = True
                
        return contains_pci