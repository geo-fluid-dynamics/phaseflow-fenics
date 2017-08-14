import fenics
import helpers

def make(form_factory, newton_relative_tolerance=1.e-8, max_newton_iterations=12, linearize=False,
        adaptive_space=False, adaptive_space_error_tolerance=1.e-4):
    ''' This function allows us to create a time solver function with a consistent interface. Among other reasons for this, the interfaces for the FEniCS classes AdaptiveLinearVariationalSolver and LinearVariationalSolver are not consistent. '''
    
    if not adaptive_space:     
            
        def solve(problem):
            
            solver = fenics.NonlinearVariationalSolver(problem)
            
            solver.parameters['newton_solver']['maximum_iterations'] = max_newton_iterations
            
            solver.parameters['newton_solver']['relative_tolerance'] = newton_relative_tolerance
        
            solver.parameters['newton_solver']['error_on_nonconvergence'] = False
        
            iteration_count, converged = solver.solve()
            
            return converged
            
    else:

        def solve(problem, M):
        
            solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

            solver.solve(adaptive_space_error_tolerance)

            solver.summary()
            
            converged = True 
            
            return converged
            

    def solve_time_step(dt, w, w_n, bcs, pci_refinement_cycles=0):

        '''  @todo Implement adaptive time for nonlinear version.
        How to get residual from solver.solve() to check if diverging? 
        Related: Set solver.parameters.nonlinear_variational_solver.newton_solver["error_on_nonconvergence"] = False and figure out how to read convergence data.'''
        F, J = form_factory.make_nonlinear_form(dt=dt, w_k=w, w_n=w_n)
        
        problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
        
        u, p, theta = fenics.split(w)
        
        if adaptive_space:
        
            M = fenics.sqrt(u[0]**2 + theta**2)*fenics.dx # @todo Handle n-D
            
            converged = solve(problem=problem, M=M)
        
        else:
        
            converged = solve(problem=problem)
            
        return converged
            
        
    return solve_time_step