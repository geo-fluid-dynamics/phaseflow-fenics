import fenics

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import phaseflow


def stefan_problem_limits_linearized():

    theta_h = 1.
    
    theta_c = -1.
    
    for Ste in (1., 0.1, 0.01):
    
        R_s = 0.
    
        w = phaseflow.run(
            output_dir = 'output/stefan_problem_limits_Ste'+str(Ste).replace('.', 'p')+'_linearized/',
            output_format = 'table',
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
            regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': 0.005},
            final_time = 0.01,
            time_step_bounds = 0.001,
            linearize = True)


if __name__=='__main__':
    
    stefan_problem_limits_linearized()
