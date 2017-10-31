from .context import phaseflow

import fenics
import scipy.optimize as opt


def verify_melting_pci_position(Ste, r, w):
    
    data = [
        {'Ste': 1., 'time': 0.01, 'true_pci_pos': 0.075551957640682}, 
        {'Ste': 0.1, 'time': 0.01, 'true_pci_pos': 0.037826726426565},
        {'Ste': 0.01, 'time': 0.1, 'true_pci_pos': 0.042772111844781}] # From Kai's MATLAB script
    
    def theta(x):
        
        wval = w(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.newton(theta, 0.075)
    
    for record in data:
        
        if Ste == record['Ste']:
            
            assert(abs(pci_pos - record['true_pci_pos']) < r)
            
            
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
        regularization = {'T_f': 0.01, 'r': r},
        nlp_relative_tolerance = 1.e-8,
        adaptive_solver_tolerance = 1.e-8,
        end_time = end_time,
        time_step_size = dt)
        
    return w
        

def test_stefan_problem_Ste1():
    
    Ste = 1.
    
    r = 0.005
    
    w = stefan_problem(Ste=Ste, r=r, dt=0.001, end_time = 0.01, initial_uniform_cell_count=1)
    
    verify_melting_pci_position(Ste, r, w)

    
def test_stefan_problem_Ste0p1():
    
    Ste = 0.1
    
    r = 0.05
    
    w = stefan_problem(Ste=Ste, r=r, dt=1.e-4, end_time = 0.01, initial_uniform_cell_count=10)
    
    verify_melting_pci_position(Ste, r, w)
    
    
""" The Ste = 0.01 case takes too long to include in the standard test suite. Maybe something reasonable
can be done with different parameters and with a different final time. The following should run,
but took almost thirty minutes on my laptop:

def test_stefan_problem_Ste0p01():

    Ste = 0.01
    
    r = 0.1
    
    w = stefan_problem(Ste=Ste, r=r, dt=1.e-4, end_time = 0.1,
            initial_uniform_cell_count=100, nlp_absolute_tolerance=0.1)
    
        verify_melting_pci_position(Ste, r, w)
"""


def verify_solidification_pci_position(w, r):
    """ Verify based on analytical model from Worster 2000 """
    data = [{'Ste': 0.125, 'time': 1., 'true_pci_pos': 0.49}] # From MATLAB script solving Worster2000
    
    def theta(x):
        
        wval = w(fenics.Point(x))
        
        return wval[2]
        
    pci_pos = opt.newton(theta, 0.1)
    
    for record in data:
        
        assert(abs(pci_pos - record['true_pci_pos']) < r)
        
        
def stefan_problem_solidify(Ste = 0.125,
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
        regularization = {'T_f': theta_f, 'r': r},
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        adaptive_solver_tolerance = 1.e-8,
        end_time = end_time,
        time_step_size = dt)
    
    return w
    
    
def test_stefan_problem_solidify():
    
    r = 0.01
    
    w = stefan_problem_solidify(r = r)
    
    verify_solidification_pci_position(w, r)
    
    
if __name__=='__main__':
    
    test_stefan_problem_Ste1()
    
    #test_stefan_problem_Ste0p1()
    
    test_stefan_problem_solidify()
    