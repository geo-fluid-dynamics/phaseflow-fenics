import fenics
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import phaseflow


def stefan_problem_limits_refined_pci():

    theta_h = 1.
    
    theta_c = -1.
    
    for Ste in (1., 0.1, 0.01):
    
        R_s = 0.005/Ste
    
        mesh = fenics.UnitIntervalMesh(int(1/Ste))
        
        hot_boundary_refinement_cycles = 10
        
        ''' Refine mesh near hot boundary
        The usual approach of using SubDomain and EdgeFunction isn't appearing to work
        in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
        '''
        for i in range(hot_boundary_refinement_cycles):
        
            cell_markers = fenics.CellFunction("bool", mesh)
            
            cell_markers.set_all(False)
            
            for cell in fenics.cells(mesh):
            
                found_hot_boundary = False
            
                for vertex in fenics.vertices(cell):
                
                    if fenics.near(vertex.x(0), 0., fenics.dolfin.DOLFIN_EPS):
                    
                        found_hot_boundary = True
                        
                if found_hot_boundary:
                
                    cell_markers[cell] = True

                    break # There should only be one such point.
                    
            mesh = fenics.refine(mesh, cell_markers)
    
        w = phaseflow.run(
            output_dir = 'output/stefan_problem_limits_refined_pci_Ste'+str(Ste).replace('.', 'p')+'/',
            output_format = 'table',
            Pr = 1.,
            Ste = Ste,
            g = [0.],
            mesh = mesh,
            max_pci_refinement_cycles = 10,
            initial_values_expression = (
                "0.",
                "0.",
                "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) "+str(theta_c)),
            boundary_conditions = [
                {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
                {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
                {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
            regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
            newton_relative_tolerance = 1.e-4/Ste,
            final_time = 0.01,
            time_step_bounds = 0.001*Ste,
            linearize = False)

        
if __name__=='__main__':
    
    stefan_problem_limits_refined_pci()
