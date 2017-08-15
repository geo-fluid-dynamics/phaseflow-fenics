import fenics


def make(form_factory, newton_relative_tolerance=1.e-8, max_newton_iterations=12, automatic_jacobian=True):
    ''' This function allows us to create a time solver function with a consistent interface. Among other reasons for this, the interfaces for the FEniCS classes AdaptiveLinearVariationalSolver and LinearVariationalSolver are not consistent. '''  
            
    def solve(problem):
        
        solver = fenics.NonlinearVariationalSolver(problem)
        
        solver.parameters['newton_solver']['maximum_iterations'] = max_newton_iterations
        
        solver.parameters['newton_solver']['relative_tolerance'] = newton_relative_tolerance
    
        solver.parameters['newton_solver']['error_on_nonconvergence'] = False
    
        iteration_count, converged = solver.solve()
        
        return converged
            

    def solve_time_step(dt, w, w_n, bcs):

        F, J = form_factory.make_nonlinear_form(dt=dt, w_=w, w_n=w_n, automatic_jacobian=automatic_jacobian)
        
        problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
        
        converged = solve(problem=problem)
            
        return converged
            
        
    return solve_time_step