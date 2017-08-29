import fenics
import phaseflow
        
def refine_pci(Ste = 0.045,
        Ra = 3.27e5,
        Pr = 56.2,
        mu_s = 1.e8,
        theta_s = 0.1,
        R_s = 0.05,
        m = 10,
        time_step_bounds = (1.e-3, 1.e-3, 1.e-3),
        initial_pci_refinement_cycles = 6,
        max_pci_refinement_cycles = 6,
        output_dir='output/refine_pci',
        start_time=0.,
        end_time=1.,
        nlp_divergence_threshold = 1.e12,
        nlp_relaxation = 1.,
        nlp_max_iterations = 30,
        restart=False,
        restart_filepath=''):

    theta_hot = 1.
    
    theta_cold = -0.1

    w, mesh = phaseflow.run(
        Ste = Ste,
        Ra = Ra,
        Pr = Pr,
        mu_s = mu_s,
        mu_l = 1.,
        mesh = fenics.UnitSquareMesh(m, m),
        time_step_bounds = time_step_bounds,
        start_time = start_time,
        end_time = end_time,
        output_times = ('all',),
        stop_when_steady = True,
        automatic_jacobian = False,
        custom_newton = True,
        regularization = {'a_s': 2., 'theta_s': theta_s, 'R_s': R_s},
        initial_pci_refinement_cycles = initial_pci_refinement_cycles,
        max_pci_refinement_cycles = max_pci_refinement_cycles,
        nlp_max_iterations = nlp_max_iterations,
        nlp_divergence_threshold = nlp_divergence_threshold,
        nlp_relaxation = nlp_relaxation,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "("+str(theta_hot)+" - "+str(theta_cold)+")*(x[0] < 0.001) + "+str(theta_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_hot), 'degree': 2,
                'location_expression': "near(x[0],  0.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_cold), 'degree': 2,
                'location_expression': "near(x[0],  1.)",
                'method': "topological"}],
        output_dir = output_dir,
        debug = True,
        restart = restart,
        restart_filepath = restart_filepath)

    return w, mesh
    
    
def run_refine_pci():

    for r in range(3):
    
        w, mesh = refine_pci(Ste = 1.,
            theta_s = 0.,
            R_s = 0.05,
            m = 1,
            end_time = 0.,
            initial_pci_refinement_cycles = r,
            max_pci_refinement_cycles = r,
            output_dir = 'output/refine_pci_'+str(r))
        
    
if __name__=='__main__':

    run_refine_pci()
