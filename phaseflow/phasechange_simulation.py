""" **phasechange_simulation.py** provides an abstract class for phase-change simulations,
including the natural and compositional convection of binary alloys. 
"""
import phaseflow
import fenics
import matplotlib
import math
import sys


class AbstractSimulation(phaseflow.simulation.AbstractSimulation):
    """ Implement the general model for phase-change coupled with natural and compositional convection.
    
    This class is abstract, because an instantiable simulation still requires 
    definitions for the mesh, initial values, and boundary conditions. """
    def __init__(self, time_order = 1, integration_measure = fenics.dx, setup_solver = True):
        
        self.temperature_rayleigh_number = fenics.Constant(1., name = "Ra_T")
        
        self.concentration_rayleigh_number = fenics.Constant(-1., name = "Ra_C")
        
        self.prandtl_number = fenics.Constant(1., name = "Pr")
        
        self.stefan_number = fenics.Constant(1., name = "Ste")
        
        self.schmidt_number = fenics.Constant(1., name = "Sc")
        
        self.pure_liquidus_temperature = fenics.Constant(0., name = "T_m")
        
        self.liquidus_slope = fenics.Constant(-1., name = "m_L")
        
        self.liquid_viscosity = fenics.Constant(1., name = "mu_L")
        
        self.solid_viscosity = fenics.Constant(1.e8, name = "mu_S")
        
        self.pressure_penalty_factor = fenics.Constant(1.e-7, name = "gamma")
        
        self.regularization_central_temperature_offset = fenics.Constant(0., name = "delta_T")
        
        self.regularization_smoothing_parameter = fenics.Constant(1./64., name = "r")
    
        self.regularization_sequence = None
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
            
        if setup_solver:
        
            self.solver.parameters["newton_solver"]["maximum_iterations"] = 12
        
            self.solver.parameters["newton_solver"]["absolute_tolerance"] = 1.e-9
    
    def phi(self, T, C, T_m, m_L, delta_T, s):
        """ The regularized semi-phasefield. """
        T_L = delta_T + T_m + m_L*C
        
        tanh = fenics.tanh
        
        return 0.5*(1. + tanh((T_L - T)/s))
        
    def semi_phasefield(self, T, C):
        """ The semi-phasefield $phi$ given in UFL. """
        T_m = self.pure_liquidus_temperature
        
        m_L = self.liquidus_slope
        
        delta_T = self.regularization_central_temperature_offset
        
        s = self.regularization_smoothing_parameter
        
        return self.phi(T = T, C = C, T_m = T_m, m_L = m_L, delta_T = delta_T, s = s)
        
    def point_value_from_semi_phasefield(self, T, C):
        """ The semi-phasefield $phi$ sutiable for evaluation given a single $T$ and $C$. 
        
        Maybe there is a way to evaluate the UFL expression rather than having to provide this
        redundant function.
        """
        T_m = self.pure_liquidus_temperature.values()[0]
        
        m_L = self.liquidus_slope.values()[0]
        
        delta_T = self.regularization_central_temperature_offset.values()[0]
        
        s = self.regularization_smoothing_parameter.values()[0]
        
        return self.phi(T = T, C = C, T_m = T_m, m_L = m_L, delta_T = delta_T, s = s)
        
    def time_discrete_terms(self):
        """ Return the discrete time derivatives which are needed for the variational form. """
        p_t, u_t, T_t, C_t = super().time_discrete_terms()
        
        pnp1, unp1, Tnp1, Cnp1_L = fenics.split(self._solutions[0].leaf_node())
        
        pn, un, Tn, Cn_L = fenics.split(self._solutions[1].leaf_node())
        
        phinp1 = self.semi_phasefield(T = Tnp1, C = Cnp1_L)
        
        phin = self.semi_phasefield(T = Tn, C = Cn_L)
        
        if self.time_order == 1:
        
            phi_t = phaseflow.backward_difference_formulas.apply_backward_euler(
                Delta_t = self._timestep_sizes[0], 
                u = (phinp1, phin))
        
        if self.time_order > 1:
        
            pnm1, unm1, Tnm1, Cnm1_L = fenics.split(self._solutions[2])
            
            phinm1 = self.semi_phasefield(T = Tnm1, C = Cnm1_L)
        
        if self.time_order == 2:
        
            phi_t = phaseflow.backward_difference_formulas.apply_bdf2(
                Delta_t = (self._timestep_sizes[0], self._timestep_sizes[1]),
                u = (phinp1, phin, phinm1))
                
        if self.time_order > 2:
            
            raise NotImplementedError()
            
        return u_t, T_t, C_t, phi_t
        
    def element(self):
        """ Return a P1P2P1P1 element for the monolithic solution. """
        P1 = fenics.FiniteElement('P', self.mesh.ufl_cell(), 1)
        
        P2 = fenics.VectorElement('P', self.mesh.ufl_cell(), 2)
        
        return fenics.MixedElement([P1, P2, P1, P1])
        
    def buoyancy(self, T, C):
        """ Extend the model from @cite{zimmerman2018monolithic} with a solute concentration. """
        Pr = self.prandtl_number
        
        Ra_T = self.temperature_rayleigh_number
        
        Ra_C = self.concentration_rayleigh_number
        
        ghat = fenics.Constant((0., -1.), name = "ghat")
        
        return 1./Pr*(Ra_T*T + Ra_C*C)*ghat
        
    def governing_form(self):
        """ Extend the model from @cite{zimmerman2018monolithic} with a solute concentration balance. """
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        gamma = self.pressure_penalty_factor
        
        mu_L = self.liquid_viscosity
        
        mu_S = self.solid_viscosity
        
        p, u, T, C = fenics.split(self.solution.leaf_node())
        
        u_t, T_t, C_t, phi_t = self.time_discrete_terms()
        
        b = self.buoyancy(T = T, C = C)
        
        phi = self.semi_phasefield(T = T, C = C)
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        psi_p, psi_u, psi_T, psi_C = fenics.TestFunctions(self.function_space)
        
        inner, dot, grad, div, sym = \
            fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        mass = -psi_p*div(u)
        
        momentum = dot(psi_u, u_t + b + dot(grad(u), u)) - div(psi_u)*p \
            + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))
        
        enthalpy = psi_T*(T_t - 1./Ste*phi_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        concentration = \
            psi_C*((1. - phi)*C_t - C*phi_t) \
            + dot(grad(psi_C), 1./Sc*(1. - phi)*grad(C) - C*u)
        
        stabilization = -gamma*psi_p*p
        
        dx = self.integration_measure
        
        F =  (mass + momentum + enthalpy + concentration + stabilization)*dx
        
        return F
    
    def adaptive_goal(self):
        """ Choose the melting rate as the goal. """
        u_t, T_t, C_t, phi_t = self.time_discrete_terms()
        
        dx = self.integration_measure
        
        return -phi_t*dx
        
    def solve_with_auto_regularization(self, 
            goal_tolerance = None,
            max_regularization_threshold = 4., 
            max_attempts = 16,
            enable_newton_solution_backup = False):
        """ Catch solver failure and automatically over-regularize the problem, 
        then successively return to desired regularization.
        
        If not using AMR, then the latest successful Newton solution can be saved/loaded to be more efficient
        with `enable_newton_solution_backup = True`.
        """
        if self.regularization_sequence == None:
        
            self.regularization_sequence = (self.regularization_smoothing_parameter.__float__(),)
        
        first_sigma_to_solve = self.regularization_sequence[0]
        
        attempts = range(max_attempts)
        
        solved = False
        
        for attempt in attempts:

            sigma_start_index = self.regularization_sequence.index(first_sigma_to_solve)
            
            try:
            
                for sigma in self.regularization_sequence[sigma_start_index:]:
                    
                    self.regularization_smoothing_parameter.assign(sigma)
                    
                    if enable_newton_solution_backup:
                    
                        self.save_newton_solution()
                    
                    self.solve(goal_tolerance = goal_tolerance)
                    
                solved = True
                
                break
                
            except RuntimeError:  
                
                if "Newton solver did not converge" not in str(sys.exc_info()):
                
                    raise
                
                current_sigma = self.regularization_smoothing_parameter.__float__()
                
                sigmas = self.regularization_sequence
                
                print("Failed to solve with sigma = " + str(current_sigma) + 
                     " from the sequence " + str(sigmas))
                
                if attempt == attempts[-1]:
                    
                    break
                    
                index = sigmas.index(current_sigma)
                
                if index == 0:
                
                    sigma_to_insert = 2.*sigmas[0]
                    
                    new_sigmas = (sigma_to_insert,) + sigmas
                
                else:
                
                    sigma_to_insert = (current_sigma + sigmas[index - 1])/2.
                
                    new_sigmas = sigmas[:index] + (sigma_to_insert,) + sigmas[index:]
                
                self.regularization_sequence = new_sigmas
                
                print("Inserted new value of " + str(sigma_to_insert))
                
                if enable_newton_solution_backup:
                
                    self.load_newton_solution()
                
                    first_sigma_to_solve = sigma_to_insert
                
                else:
                
                    self.reset_initial_guess()
                    
                    first_sigma_to_solve = self.regularization_sequence[0]
        
        self.regularization_smoothing_parameter.assign(self.regularization_sequence[-1])
        
        assert(solved)
        
    def coarsen(self, 
            absolute_tolerances = (1.e-2, 1.e-2, 1.e-2, 1.e-2, 1.e-2),
            maximum_refinement_cycles = 6, 
            circumradius_threshold = 0.01):
        """ Re-mesh while preserving pointwise accuracy of solution variables. """
        finesim = self.deepcopy()
        
        adapted_coares_mesh = self.coarse_mesh()
        
        adapted_coarse_function_space = fenics.FunctionSpace(adapted_coares_mesh, self._element)
        
        adapted_coarse_solution = fenics.Function(adapted_coarse_function_space)
        
        assert(self.mesh.topology().dim() == 2)
        
        def u0(solution, point):
        
            return solution(point)[1]
            
        def u1(solution, point):
        
            return solution(point)[2]
            
        def T(solution, point):
        
            return solution(point)[3]
        
        def C(solution, point):
        
            return solution(point)[4]
        
        def phi(solution, point):
            
            return self.point_value_from_semi_phasefield(T = T(solution, point), C = C(solution, point))
        
        scalars = (u0, u1, T, C, phi)
        
        for scalar, tolerance in zip(scalars, absolute_tolerances):
        
            adapted_coarse_solution, adapted_coarse_function_space, adapted_coarse_mesh = \
                phaseflow.refinement.adapt_coarse_solution_to_fine_solution(
                    scalar = scalar,
                    coarse_solution = adapted_coarse_solution,
                    fine_solution = finesim.solution,
                    element = self._element,
                    absolute_tolerance = tolerance, 
                    maximum_refinement_cycles = maximum_refinement_cycles, 
                    circumradius_threshold = circumradius_threshold)
                    
        self._mesh = adapted_coarse_mesh
        
        self._function_space = fenics.FunctionSpace(self._mesh, self._element)
        
        for i in range(len(self._solutions)):
            
            self._solutions[i] = fenics.project(
                finesim._solutions[i].leaf_node(), self._function_space.leaf_node())
        
        self.setup_solver()
        
    def deepcopy(self):
        """ Extends the parent deepcopy method with attributes for this derived class """
        sim = super().deepcopy()
        
        sim.temperature_rayleigh_number.assign(self.temperature_rayleigh_number)
        
        sim.concentration_rayleigh_number.assign(self.concentration_rayleigh_number)
        
        sim.prandtl_number.assign(self.prandtl_number)
        
        sim.stefan_number.assign(self.stefan_number)
        
        sim.schmidt_number.assign(self.schmidt_number)
        
        sim.pure_liquidus_temperature.assign(self.pure_liquidus_temperature)
        
        sim.liquidus_slope.assign(self.liquidus_slope)
        
        sim.liquid_viscosity.assign(self.liquid_viscosity)
        
        sim.solid_viscosity.assign(self.solid_viscosity)
        
        sim.pressure_penalty_factor.assign(self.pressure_penalty_factor)
        
        sim.regularization_central_temperature_offset.assign(
            self.regularization_central_temperature_offset)
        
        sim.regularization_smoothing_parameter.assign(self.regularization_smoothing_parameter)
        
        return sim
        
    def _plot(self, solution, time, savefigs = False, outdir = ""):
        """ Plot the adaptive mesh, velocity vector field, temperature field, and phase field. """
        p, u, T, C = solution.leaf_node().split()
        
        phi = fenics.project(self.semi_phasefield(T = T, C = C), mesh = self.mesh.leaf_node())
        
        C = fenics.project(C*(1. - phi), mesh = self.mesh.leaf_node())
       
        for var, label, colorbar, varname in zip(
                (solution.function_space().mesh().leaf_node(), u, T, C, phi),
                ("$\Omega_h$", "$\mathbf{u}$", "$T$", "$C$", "$\phi(T,C)$"),
                (False, True, True, True, True),
                ("mesh", "u", "T", "C", "phi")):
            
            some_mappable_thing = phaseflow.plotting.plot(var)
            
            if colorbar and (self.mesh.topology().dim() > 1):
            
                matplotlib.pyplot.colorbar(some_mappable_thing)
            
            matplotlib.pyplot.title(label + ", $t = " + str(time) + "$")
            
            matplotlib.pyplot.xlabel("$x$")
            
            if colorbar and (self.mesh.topology().dim() > 1):
            
                matplotlib.pyplot.ylabel("$y$")
            
            if savefigs:
            
                matplotlib.pyplot.savefig(outdir + "/" + varname + "_t" + str(time) + ".png")
            
            matplotlib.pyplot.show()
    
    def write_solution(self, file, solution_index = 0):
        """ Write the solution to a file.
        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile
            This method should have been called from within the context of the open `file`.
        """
        for var, symbol, label in zip(
                self._solutions[solution_index].leaf_node().split(), 
                ("p", "u", "T", "C"), 
                ("pressure", "velocity", "temperature", "concentration")):
        
            var.rename(symbol, label)
            
            file.write(var, self._times[solution_index])
