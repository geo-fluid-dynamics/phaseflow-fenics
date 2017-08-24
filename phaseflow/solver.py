import fenics

import phaseflow.problem


class Solver(fenics.NewtonSolver):

    custom_parameters = {'divergence_threshold': 1.e8}

    def converged(self, residual, problem, iteration):
    
        rnorm = residual.norm("l2")
        
        print("Iteration %d: L2 norm of residual is %.3e" % (iteration, rnorm))
        
        assert(rnorm < self.custom_parameters['divergence_threshold'])
        
        if rnorm < self.parameters['absolute_tolerance']:
            
            return True
            
        return False


def make(form_factory,
        nlp_absolute_tolerance=1., nlp_relative_tolerance=1.e-8, nlp_max_iterations=12,
        nlp_method='newton', 
        automatic_jacobian=True,):
    ''' This function allows us to create a time solver function with a consistent interface. Among other reasons for this, the interfaces for the FEniCS classes AdaptiveLinearVariationalSolver and LinearVariationalSolver are not consistent. '''  
            
    def solve(problem, w):
        
        solver = Solver()
        
        solver.parameters['maximum_iterations'] = nlp_max_iterations
        
        solver.parameters['absolute_tolerance'] = nlp_absolute_tolerance
        
        solver.parameters['relative_tolerance'] = nlp_relative_tolerance
    
        solver.parameters['error_on_nonconvergence'] = False
    
        iteration_count, converged = solver.solve(problem, w.vector())
        
        return converged
            

    def solve_time_step(dt, w, w_n, bcs):

        F, J = form_factory.make_nonlinear_form(dt=dt, w_k=w, w_n=w_n, automatic_jacobian=automatic_jacobian)
        
        problem = phaseflow.problem.Problem(J, F, bcs)
        
        converged = solve(problem=problem, w=w)
            
        return converged
            
        
    return solve_time_step