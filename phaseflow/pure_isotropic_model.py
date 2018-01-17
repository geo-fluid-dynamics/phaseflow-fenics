import phaseflow.pure_material


class PureIsotropicProblem(phaseflow.ImplicitEulerIBVP):

    def __init__(self,
            state,
            boundary_conditions,
            buoyancy_model,
            semi_phasefield_mapping,
            time_step_size = 1.,
            rayleigh_number = 1.,
            prandtl_number = 1.,
            stefan_number = 1.,
            liquid_viscosity = 1.,
            solid_viscosity = 1.e8,
            gravity = (0., -1.),
            penalty_parameter = 1.e-7,
            integration_metric = fenics.dx)
        """
        Parameters
        ----------
        state : phaseflow.State
        
        time_step_size : float
        
        boundary_conditions : [fenics.DirichletBoundaryCondition,]
        
        buoyancy_model : phaseflow.ContinuousFunctionOfTemperature
        
        semi_phasefield_mapping : phaseflow.ContinuousFunctionOfTemperature
        """
        
        
        ## Set the variational form.
        """Set local names for math operators to improve readability."""
        inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        """The linear, bilinear, and trilinear forms b, a, and c, follow the common notation 
        for applying the finite element method to the incompressible Navier-Stokes equations,
        e.g. from danaila2014newton and huerta2003fefluids.
        """
        def b(u, p):
            return -div(u)*p  # Divergence
        
        
        def D(u):
        
            return sym(grad(u))  # Symmetric part of velocity gradient
        
        
        def a(mu, u, v):
            
            return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
        
        
        def c(u, z, v):
            
            return dot(dot(grad(z), u), v)  # Convection of the velocity field
        
        
        Delta_t = fenics.Constant(time_step_size)
        
        Pr = fenics.Constant(prandtl_number)
        
        Ste = fenics.Constant(stefan_number)
        
        g = fenics.Constant(gravity)
        
        f_B = buoyancy_model.function
        
        phi = semi_phasefield_mapping.function
        
        gamma = fenics.Constant(penalty_parameter)
        
        mu_L = fenics.Constant(liquid_viscosity)
        
        mu_S = fenics.Constant(solid_viscosity)
        
        phase_dependent_viscosity = phaseflow.pure_material.PhaseDependentMaterialProperty(mu_L, mu_S)
        
        mu = phase_dependent_viscosity.function
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(W)
        
        w = state.solution
        
        p, u, T = fenics.split(w)
        
        w_n = old_state.solution
         
        p_n, u_n, T_n = fenics.split(w_n)
        
        dx = integration_metric
        
        F = (
            b(u, psi_p) - psi_p*gamma*p
            + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
            + c(u, u, psi_u) + b(psi_u, p) + a(mu(T), u, psi_u)
            + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)        
            )*dx

            
        # Set the Gateaux derivative in variational form.
        df_B = buoyancy_model.derivative_function
        
        dphi = semi_phasefield_mapping.derivative_function
        
        dmu = phase_dependent_viscosity.function_derivative
            
        delta_w = fenics.TrialFunction(W)
        
        delta_p, delta_u, delta_T = fenics.split(delta_w)
        
        w_k = w
        
        p_k, u_k, T_k = fenics.split(w_k)
        
        gateaux_derivative = (
            b(delta_u, psi_p) - psi_p*gamma*delta_p 
            + dot(psi_u, 1./Delta_t*delta_u + delta_T*ddT_f_B(T_k))
            + c(u_k, delta_u, psi_u) + c(delta_u, u_k, psi_u) 
            + b(psi_u, delta_p) 
            + a(delta_T*dmu(T_k), u_k, psi_u) + a(mu(T_k), delta_u, psi_u) 
            + 1./Delta_t*psi_T*delta_T*(1. - 1./Ste*dphi(T_k))
            + dot(grad(psi_T), 1./Pr*grad(delta_T) - T_k*delta_u - delta_T*u_k)
            )*fenics.dx

            
        # Construct the initial boundary value problem.
        phaseflow.ImplicitEulerIBVP.__init__(self, 
            nonlinear_variational_form, 
            state, 
            boundary_conditions, 
            gateaux_derivative)
    