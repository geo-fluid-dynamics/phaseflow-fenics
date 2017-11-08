from .context import phaseflow

import fenics
import scipy.optimize as opt

""" Melting data From Kai's MATLAB script"""
melting_data = [
    {'Ste': 1., 'time': 0.01, 'true_pci_pos': 0.075551957640682}, 
    {'Ste': 0.1, 'time': 0.01, 'true_pci_pos': 0.037826726426565},
    {'Ste': 0.01, 'time': 0.1, 'true_pci_pos': 0.042772111844781}] 
    
"""Solidification datat from MATLAB script solving Worster2000"""
solidification_data = [{'Ste': 0.125, 'time': 1., 'true_pci_pos': 0.49}] 

def verify_pci_position(true_pci_position, r, w):
    
    def theta(x):
        
        wval = w.leaf_node()(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.newton(theta, 0.1)
    
    assert(abs(pci_pos - true_pci_position) < r)
    

def refine_near_left_boundary(mesh, cycles):
    """ Refine mesh near the left boundary.
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    """
    for i in range(cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
                break # There should only be one such point.
                
        mesh = fenics.refine(mesh, cell_markers)
        
    return mesh
    
    
def stefan_problem(Ste = 1.,
        theta_h = 1.,
        theta_c = -1.,
        r = 0.005,
        dt = 0.001,
        end_time = 0.01,
        initial_uniform_cell_count = 1,
        hot_boundary_refinement_cycles = 10):
    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    mesh = refine_near_left_boundary(mesh, hot_boundary_refinement_cycles)
    
    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_Ste'+str(Ste).replace('.', 'p')+'/',
        prandtl_number = 1.,
        stefan_number = Ste,
        gravity = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) + "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = 0.01,
        regularization_smoothing_factor = r,
        nlp_relative_tolerance = 1.e-8,
        adaptive = True,
        adaptive_solver_tolerance = 1.e-8,
        end_time = end_time,
        time_step_size = dt)
        
    return w
        

def test_stefan_problem_Ste1():
    
    Ste = 1.
    
    r = 0.005
    
    w = stefan_problem(Ste=Ste, r=r, dt=0.001, end_time = 0.01, initial_uniform_cell_count=1)
    
    """ Verify against solution from Kai's MATLAB script. """
    verify_pci_position(true_pci_position=0.0755, r=r, w=w)

    
def test_stefan_problem_Ste0p01__nightly():

    Ste = 0.01
    
    r = 0.1
    
    w = stefan_problem(Ste=Ste, r=r, dt=1.e-4, end_time = 0.1, initial_uniform_cell_count=100)
    
    """ Verify against solution from Kai's MATLAB script. """
    verify_pci_position(true_pci_position=0.04277, r=r, w=w)


def test_stefan_problem_solidify__nightly(Ste = 0.125,
        theta_h = 0.01,
        theta_c = -1.,
        theta_f = 0.,
        r = 0.01,
        dt = 0.01,
        end_time = 1.,
        nlp_absolute_tolerance = 1.e-4,
        initial_uniform_cell_count = 100,
        cool_boundary_refinement_cycles = 0):
    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    mesh = refine_near_left_boundary(mesh, cool_boundary_refinement_cycles)
    
    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_solidify/dt'+str(dt)+
            '/tol'+str(nlp_absolute_tolerance)+'/',
        prandtl_number = 1.,
        stefan_number = Ste,
        gravity = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_c)+" - "+str(theta_h)+")*near(x[0],  0.) + "+str(theta_h)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = theta_f,
        regularization_smoothing_factor = r,
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        adaptive = True,
        adaptive_solver_tolerance = 1.e-8,
        end_time = end_time,
        time_step_size = dt)
    
    """ Verify against solution from MATLAB script solving Worster2000. """
    verify_pci_position(true_pci_position=0.49, r=r, w=w)

    
if __name__=='__main__':
    
    test_stefan_problem_Ste1()
    
    test_stefan_problem_Ste0p01__nightly()
    
    test_stefan_problem_solidify__nightly()
    