from .context import phaseflow

import fenics
import scipy.optimize as opt


def verify_pci_position(Ste, R_s, w):

    data = [
        {'Ste': 1., 'true_pci_pos':    0.075551957640682}, 
        {'Ste': 0.1, 'true_pci_pos':    0.037826726426565}] # From Kai's MATLAB script
    
    def theta(x):
    
        wval = w(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.newton(theta, 0.075)
    
    for record in data:
    
        if Ste == record['Ste']:
        
            assert(abs(pci_pos - record['true_pci_pos']) < R_s)
        
        
def test_stefan_problem():

    theta_h = 1.
    
    theta_c = -1.
    
    R_s = 0.005
    
    for Ste in [1., 0.1]:
    
        w = phaseflow.run(
            write_output = False,
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
                {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
                {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
            regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
            final_time = 0.01,
            time_step_bounds = 0.001,
            linearize = False)

        verify_pci_position(Ste, R_s, w)
    

if __name__=='__main__':
    
    test_stefan_problem()
