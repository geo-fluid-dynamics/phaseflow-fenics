"""This module contains routines to make the solver."""
import fenics


class Problem(fenics.NonlinearProblem):
    """This derived class is required for the derived Newton solver class."""
    def __init__(self, a, L, bcs):
    
        fenics.NonlinearProblem.__init__(self)
        
        self.a = a
        
        self.L = L
        
        self.bcs = bcs
    
    def F(self, b, x):

        assembler = fenics.SystemAssembler(self.a, self.L, self.bcs)
        
        assembler.assemble(b, x)
        
        
    def J(self, A, x):

        assembler = fenics.SystemAssembler(self.a, self.L, self.bcs)
        
        assembler.assemble(A)
        

def make(form_factory,
        nlp_absolute_tolerance=1.,
        nlp_relative_tolerance=1.e-8,
        nlp_max_iterations=12,
        nlp_divergence_threshold=1.e12,
        nlp_relaxation=1.,
        custom_newton=True,
        automatic_jacobian=True):
    """ Create a time solver function with a consistent interface.
    
    Among other reasons for this, the interfaces for the FEniCS classes 
    AdaptiveLinearVariationalSolver and LinearVariationalSolver are not
    consistent. 
    """
    
    """@todo This may be deprecated 
    now that we always use a nonlinear solver and 
    also since we no longer any of FEniCS's adaptive solvers,
    since their adaptive solvers do not work in parallel.
    """        

    class CustomNewtonSolver(fenics.NewtonSolver):
        """This derived class allows us to catch divergence."""
        custom_parameters = {'divergence_threshold': nlp_divergence_threshold}

        def converged(self, residual, problem, iteration):
        
            rnorm = residual.norm("l2")
            
            print("Newton iteration %d: r (abs) = %.3e (tol = %.3e)" % (iteration, rnorm,
                self.parameters['absolute_tolerance']))
            
            assert(rnorm < self.custom_parameters['divergence_threshold'])
            
            if rnorm < self.parameters['absolute_tolerance']:
                
                return True
                
            return False
    
    
    '''
    Currently we have the option for which Newton solver to use for two reasons:
    1. I don't know how to get the relative residual in CustomNewtonSolver.
    2. The wang tests are failing since adding the CustomNewtonSolver (among other changes). As of
       this writing, the tests also fail with the original Newton solver.
    '''
    if custom_newton:
    
        def solve(problem, w):
        
            solver = CustomNewtonSolver()
        
            solver.parameters['maximum_iterations'] = nlp_max_iterations
        
            solver.parameters['absolute_tolerance'] = nlp_absolute_tolerance
            
            solver.parameters['relative_tolerance'] = nlp_relative_tolerance
        
            solver.parameters['error_on_nonconvergence'] = False
            
            solver.set_relaxation_parameter(nlp_relaxation)
        
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
        """Solve the problem for one time step."""
        
        F, J = form_factory.make_nonlinear_form(dt=dt, w_k=w, w_n=w_n, automatic_jacobian=automatic_jacobian)
        
        if custom_newton:
        
            problem = Problem(J, F, bcs)
            
            converged = solve(problem=problem, w=w)
            
        else:

            problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
            
            converged = solve(problem=problem)
            
        return converged
            
        
    return solve_time_step
    
    
if __name__=='__main__':

    pass
    