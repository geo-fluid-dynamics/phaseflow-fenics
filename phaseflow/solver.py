import fenics

def make(form_factory, linearize=False, adaptive_space=False, adaptive_space_error_tolerance=1.e-4):
    ''' This function allows us to create a time solver function with a consistent interface. Among other reasons for this, the interfaces for the FEniCS classes AdaptiveLinearVariationalSolver and LinearVariationalSolver are not consistent. '''
    
    if linearize:
    
        '''@todo Encapsulate this Newton method '''
        MAX_NEWTON_ITERATIONS = 12 
        
        NEWTON_RELATIVE_TOLERANCE = 1.e-9
        
        '''w_w was previously a TrialFunction, but must be a Function when defining M and when calling solve(). The details here are opaque to me. Here is a related issue: https://fenicsproject.org/qa/12271/adaptive-stokes-perform-compilation-unable-extract-indices'''
        w_w = fenics.Function(form_factory.W)

        u_w, p_w, theta_w = fenics.split(w_w)
        
        if not adaptive_space:
            
            def solve(A, L, w_w, bcs, M):
            
                fenics.solve(A == L, w_w, bcs=bcs, M=M)
            
        else:
                
            M = fenics.sqrt((u_k[0] - u_w[0])**2 + (u_k[1] - u_w[1])**2 + (theta_k - theta_w)**2)*fenics.dx
                
            def solve(A, L, w_w, bcs, M):
            
                problem = fenics.LinearVariationalProblem(A, L, w_w, bcs=bcs)

                solver = fenics.AdaptiveLinearVariationalSolver(problem, M)

                solver.solve(adaptive_space_error_tolerance)

                solver.summary()
                
    
        def solve_time_step(dt, w, w_n, bcs):
            
            print '\nIterating Newton method'
            
            converged = False
            
            iteration_count = 0
            
            w_k = w_n.copy(deepcopy=True)
            
            u_k, p_k, theta_k = fenics.split(w_k)
            
            for k in range(MAX_NEWTON_ITERATIONS):
            
                A, L = form_factory.make_newton_linearized_form(dt=dt, w_n=w_n, w_k=w_k)
                
                # Adaptive mesh refinement metric
                M = fenics.sqrt((u_k[0] - u_w[0])**2 + (u_k[1] - u_w[1])**2 + (theta_k - theta_w)**2)*fenics.dx

                solve(A=A, L=L, w_w=w_w, bcs=bcs, M=M)

                w_k.assign(w_k - w_w)
                
                norm_residual = fenics.norm(w_w, 'L2')/fenics.norm(w_k, 'L2')

                print '\nL2 norm of relative residual, || w_w || / || w_k || = ' + str(norm_residual) + '\n'
                
                if norm_residual < NEWTON_RELATIVE_TOLERANCE:
                    
                    iteration_count = k + 1
                    
                    print 'Converged after ' + str(k) + ' iterations'
                    
                    converged = True
                    
                    break
                     
            w.assign(w_k)
            
            return converged
    
    else:
        
        if not adaptive_space:     
                
            def solve(problem):
                
                solver = fenics.NonlinearVariationalSolver(problem)
            
                iteration_count, converged = solver.solve()
                
                return converged
                
        else:
    
            def solve(problem, M):
            
                solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

                solver.solve(adaptive_space_error_tolerance)

                solver.summary()
                
                converged = True 
                
                return converged
                
    
        def solve_time_step(dt, w, w_n, bcs):
    
            '''  @todo Implement adaptive time for nonlinear version.
            How to get residual from solver.solve() to check if diverging? 
            Related: Set solver.parameters.nonlinear_variational_solver.newton_solver["error_on_nonconvergence"] = False and figure out how to read convergence data.'''
            F = form_factory.make_nonlinear_form(dt=dt, w=w, w_n=w_n)
            
            J = fenics.derivative(F, w)
            
            problem = fenics.NonlinearVariationalProblem(F, w, bcs, J)
            
            u, p, theta = fenics.split(w)
            
            if adaptive_space:
            
                M = fenics.sqrt(u[0]**2 + u[1]**2 + theta**2)*fenics.dx
                
                converged = solve(problem=problem, M=M)
            
            else:
            
                converged = solve(problem=problem)
                
            return converged
            
        
    return solve_time_step