from .context import phaseflow

import fenics
import scipy.optimize as opt


def verify_pci_position(Ste, R_s, w):

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
        
            assert(abs(pci_pos - record['true_pci_pos']) < R_s)
        
        
def stefan_problem(Ste = 1.,
    theta_h = 1.,
    theta_c = -1.,
    a_s = 2.,
    R_s = 0.005,
    dt = 0.001,
    final_time = 0.01,
    newton_relative_tolerance = 1.e-3,
    initial_uniform_cell_count = 1,
    hot_boundary_refinement_cycles = 10,
    max_pci_refinement_cycles = 10):

    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
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

    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_Ste'+str(Ste).replace('.', 'p')+'/',
        output_format = 'table',
        Pr = 1.,
        Ste = Ste,
        g = [0.],
        mesh = mesh,
        max_pci_refinement_cycles = max_pci_refinement_cycles,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
        newton_relative_tolerance = newton_relative_tolerance,
        final_time = final_time,
        time_step_bounds = dt,
        output_times = (),
        linearize = False)
        
    return w
        
        
def test_stefan_problem_vary_Ste():

    for p in [{'Ste': 1., 'R_s': 0.005, 'dt': 0.001, 'final_time': 0.01, 'initial_uniform_cell_count': 1, 'newton_relative_tolerance': 1.e-3},
        {'Ste': 0.1, 'R_s': 0.05, 'dt': 0.0001, 'final_time': 0.01, 'initial_uniform_cell_count': 10, 'newton_relative_tolerance': 1.e-4}]:
        ''' The Ste = 0.01 case takes too long to include in the standard test suite. Maybe something reasonable
        can be done with different parameters and with a different final time. The following should run,
        but took almost thirty minutes on my laptop:
        {'Ste': 0.01, 'R_s': 0.1, 'dt': 0.0001, 'final_time': 0.1, 'initial_uniform_cell_count': 100, 'newton_relative_tolerance': 1.e-4}]'''
        
        w = stefan_problem(Ste=p['Ste'], R_s=p['R_s'], dt=p['dt'], final_time = p['final_time'],
            initial_uniform_cell_count=p['initial_uniform_cell_count'], newton_relative_tolerance=p['newton_relative_tolerance'])
    
        verify_pci_position(p['Ste'], p['R_s'], w)
       
       
def test_stefan_problem_linearized():

    theta_h = 1.
    
    theta_c = -1.
    
    R_s = 0.005

    for Ste in [1., 0.1]:
    
        w, mesh = phaseflow.run(
            Pr = 1.,
            Ste = Ste,
            g = [0.],
            mesh = fenics.UnitIntervalMesh(1000),
            initial_values_expression = (
                "0.",
                "0.",
                "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) "+str(theta_c)),
            boundary_conditions = [
                {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
                {'subspace': 2, 'value_expression': 0., 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
                {'subspace': 2, 'value_expression': 0., 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
            regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
            final_time = 0.01,
            time_step_bounds = 0.001,
            output_times = (),
            linearize = True)

        verify_pci_position(Ste, R_s, w)
        
        
def test_boundary_refinement():

    theta_h = 1.
    
    theta_c = -1.
    
    R_s = 0.005
    
    mesh = fenics.UnitIntervalMesh(10)
    
    hot_boundary_refinement_cycles = 6

    
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
    
    #
    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_refine_boundary/',
        output_format = 'table',
        Pr = 1.,
        Ste = 1.,
        g = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
        newton_relative_tolerance = 1.e-4,
        final_time = 0.002,
        time_step_bounds = 0.001,
        output_times = (),
        linearize = False)
        
        
def test_pci_refinement():

    Ste = 1.
    
    theta_h = 1.
    
    theta_c = -1.
    
    R_s = 0.005
    
    mesh = fenics.UnitIntervalMesh(1)
    
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
    
    #
    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_refine_pci/',
        output_format = 'table',
        Pr = 1.,
        Ste = Ste,
        g = [0.],
        mesh = mesh,
        max_pci_refinement_cycles = 3,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*near(x[0],  0.) "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
        newton_relative_tolerance = 1.e-3,
        final_time = 0.01,
        time_step_bounds = 0.001,
        output_times = (),
        linearize = False)
        
    verify_pci_position(Ste, R_s, w)
        
        
if __name__=='__main__':
    
    test_stefan_problem_vary_Ste()
    
    test_stefan_problem_linearized()

    test_boundary_refinement()
    
    test_stefan_problem_pci_refinement()