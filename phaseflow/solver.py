"""This module contains routines to make the solver."""
import fenics
import phaseflow.helpers
import phaseflow.bounded_value

MAX_RELAXATION_ATTEMPTS = 10

RELAXATION_INCREMENT = 0.1

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

        
class NewtonDiverged(Exception):
    pass
    

class Relaxation(phaseflow.bounded_value.BoundedValue):
    """This class sets bounds on the adaptive time step size."""
    def __init__(self, bounded_value):
    
        super(Relaxation, self).__init__(bounded_value.min, bounded_value.value, bounded_value.max)

    
    def set(self, value):
    
        assert(value > 0.1)

        old_value = self.value
        
        super(Relaxation, self).set(value)
        
        if abs(self.value - old_value) > fenics.DOLFIN_EPS:
        
            phaseflow.helpers.print_once("Set Newton relaxation to "
                +str(value))
    
        
def make(form_factory,
        nlp_absolute_tolerance=1.,
        nlp_relative_tolerance=1.e-8,
        nlp_max_iterations=12,
        nlp_divergence_threshold=1.e12,
        nlp_relaxation_bounds=phaseflow.bounded_value.BoundedValue(0.1,
            1., 1.),
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
    
    nlp_relaxation = Relaxation(phaseflow.bounded_value.BoundedValue(   
        nlp_relaxation_bounds[0], nlp_relaxation_bounds[1],
        nlp_relaxation_bounds[2]))

    class CustomNewtonSolver(fenics.NewtonSolver):
        """This derived class allows us to catch divergence."""
        custom_parameters = {'divergence_threshold': nlp_divergence_threshold}

        def converged(self, residual, problem, iteration):
        
            rnorm = residual.norm("l2")
            
            print("Newton iteration %d: r (abs) = %.3e (tol = %.3e)" % (iteration, rnorm,
                self.parameters['absolute_tolerance']))
            
            
            if rnorm > self.custom_parameters['divergence_threshold']:
            
                raise NewtonDiverged()
            
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
            
            solver.set_relaxation_parameter(nlp_relaxation.value)
            
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
            
            for ir in range(MAX_RELAXATION_ATTEMPTS):
            
                try:
                
                    converged = solve(problem=problem, w=w)
                    
                    if converged:
                    
                        nlp_relaxation.set(nlp_relaxation.value
                            + RELAXATION_INCREMENT)
                        
                        break
                    
                except NewtonDiverged:
                
                    if (nlp_relaxation.value < nlp_relaxation.min + fenics.DOLFIN_EPS):
                    
                        break
                        
                    nlp_relaxation.set(nlp_relaxation.value
                        - RELAXATION_INCREMENT)
                        
                    w.assign(w_n)
                    
                    continue
            
        else:

            problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
            
            converged = solve(problem=problem)
            
        return converged
            
        
    return solve_time_step
    
    
if __name__=='__main__':

    pass
    