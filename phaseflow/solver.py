import fenics

import phaseflow.problem


class CustomNewtonSolver(fenics.NewtonSolver):

    custom_parameters = {'divergence_threshold': 1.e12}

    def converged(self, residual, problem, iteration):
    
        rnorm = residual.norm("l2")
        
        print("Newton iteration %d: r (abs) = %.3e (tol = %.3e)" % (iteration, rnorm,
            self.parameters['absolute_tolerance']))
        
        assert(rnorm < self.custom_parameters['divergence_threshold'])
        
        if rnorm < self.parameters['absolute_tolerance']:
            
            return True
            
        return False

        
''' Create a time solver function with a consistent interface.
Among other reasons for this, the interfaces for the FEniCS classes 
AdaptiveLinearVariationalSolver and LinearVariationalSolver are not consistent. '''  
def make(form_factory,
        newton_absolute_tolerance = 1.,
        newton_relative_tolerance = 1.e-8,
        newton_max_iterations = 12,
        newton_custom_convergence = True,
        automatic_jacobian = True):        

    '''
    Currently we have the option for which Newton solver to use for two reasons:
    1. I don't know how to get the relative residual in CustomNewtonSolver.
    2. The wang tests are failing since adding the CustomNewtonSolver (among other changes). As of
       this writing, the tests also fail with the original Newton solver.
    '''
    if newton_custom_convergence:
    
        def solve(problem, w):
        
            solver = CustomNewtonSolver()
        
            solver.parameters['maximum_iterations'] = newton_max_iterations
        
            solver.parameters['absolute_tolerance'] = newton_absolute_tolerance
            
            solver.parameters['relative_tolerance'] = newton_relative_tolerance
        
            solver.parameters['error_on_nonconvergence'] = False
        
            iteration_count, converged = solver.solve(problem, w.vector())
            
            return converged
            
        
    else:
    
        def solve(problem):
            
            solver = fenics.NonlinearVariationalSolver(problem)
    
            solver.parameters['newton_solver']['maximum_iterations'] = nlp_max_iterations
            
            solver.parameters['newton_solver']['relative_tolerance'] = nlp_relative_tolerance
        
            solver.parameters['newton_solver']['error_on_nonconvergence'] = False
        
            iteration_count, converged = solver.solve()
            
            return converged
            
        
    def solve_time_step(dt, w, w_n, bcs):

        F, J = form_factory.make_nonlinear_form(dt=dt, w_k=w, w_n=w_n, automatic_jacobian=automatic_jacobian)
        
        if newton_custom_convergence:
        
            problem = phaseflow.problem.Problem(J, F, bcs)
            
            converged = solve(problem=problem, w=w)
            
        else:

            problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
            
            converged = solve(problem=problem)
            
        return converged
            
        
    return solve_time_step