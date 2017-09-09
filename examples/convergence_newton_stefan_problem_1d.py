import fenics
import phaseflow
import scipy.optimize as opt
    
    
def stefan_problem_solidify(Ste = 0.125,
    theta_h = 0.01,
    theta_c = -1.,
    theta_s = 0.,
    a_s = 2.,
    r = 0.01,
    dt = 0.01,
    end_time = 1.,
    nlp_absolute_tolerance = 1.e-8,
    initial_uniform_cell_count = 100,
    automatic_jacobian = False):

    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)

    w, mesh = phaseflow.run(
        output_dir = 'output/test_stefan_problem_solidify/dt'+str(dt)+
            '/tol'+str(nlp_absolute_tolerance)+'/',
        Pr = 1.,
        Ste = Ste,
        g = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_c)+" - "+str(theta_h)+")*near(x[0],  0.) + "+str(theta_h)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'a_s': a_s, 'theta_s': theta_s, 'R_s': r},
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        end_time = end_time,
        time_step_bounds = dt,
        automatic_jacobian = automatic_jacobian)
        
    return w
  

def convergence_newton_stefan_problem_1d():
    
    w = stefan_problem_solidify(end_time = 0.02)

            
if __name__=='__main__':
    
    convergence_newton_stefan_problem_1d()
