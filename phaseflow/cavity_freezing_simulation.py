import phaseflow
import fenics


class CavityFreezingSimulation(
        phaseflow.abstract_heated_cavity_phasechange_simulation.AbstractHeatedCavityPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            uniform_gridsize = 20,
            setup_solver = True):
        
        self.uniform_gridsize = uniform_gridsize
        
        self.cold_wall_temperature_before_freezing = fenics.Constant(0.1)
        
        self.cold_wall_temperature_during_freezing = fenics.Constant(-1.1)
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver,
            initial_uniform_gridsize = uniform_gridsize)
        
        self.hot_wall_temperature.assign(1.25)
        
        self.cold_wall_temperature.assign(0.25)
        
        if setup_solver:
        
            self.solver.parameters["newton_solver"]["maximum_iterations"] = 12
        
    def initial_mesh(self):
    
        return self.coarse_mesh()
    
    def initial_values(self):
    
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", 
                 "0.", 
                 "0.", 
                 "(T_c - T_h)*x[0] + T_h", 
                 "C_0"),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                C_0 = self.initial_concentration,
                element = self.element()),
            self.function_space)
            
        return initial_values
        
    def set_constant_concentration(self, value):
        
        function_space_copy = fenics.FunctionSpace(self.mesh, self.element())

        new_solution = fenics.Function(function_space_copy)

        new_solution.vector()[:] = self.solution.vector()
        
        class WholeDomain(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return True
                
        hack = fenics.DirichletBC(
            function_space_copy.sub(3),
            value,
            WholeDomain())

        hack.apply(new_solution.vector())

        for solution in self._solutions:
            
            solution.vector()[:] = new_solution.vector()
        
        for i in range(len(self._times)):
            
            self._times[i] = 0.
        
    def solve_steady_state_heat_driven_cavity(
            self,
            steady_q_tolerance = 0.01, 
            max_timesteps = 20):
        
        original_timestep_size = self.timestep_size.__float__()
        
        self.cold_wall_temperature.assign(self.cold_wall_temperature_before_freezing)
        
        self.regularization_central_temperature_offset.assign(-1.)
        
        self.assign_initial_values()
        
        self.set_constant_concentration(0.)
        
        self.timestep_size.assign(1.e-3)
        
        steady = False
        
        q_old = 1.e32
        
        for it in range(max_timesteps):

            self.solve()
            
            self.advance()
            
            q = fenics.assemble(self.cold_wall_heat_flux_integrand())
            
            if (it > 3) and (abs(q - q_old) <= steady_q_tolerance):

                steady = True

                break
                
            q_old = 0. + q
            
            self.timestep_size.assign(2.*self.timestep_size.__float__())

        assert(steady)
        
        self.set_constant_concentration(self.initial_concentration)
        
        self.timestep_size.assign(original_timestep_size)
        
    def setup_freezing_problem(self):
        
        self.cold_wall_temperature.assign(self.cold_wall_temperature_during_freezing)
        
        self.regularization_central_temperature_offset.assign(0.)
        
    def run(self, endtime = 1., checkpoint_times = (0., 1.), max_regularization_attempts = 16, 
            plot = False, savefigs = False):
        
        phaseflow.helpers.mkdir_p(self.output_dir)
        
        assert(self.temperature_rayleigh_number.__float__() > 1.e-8)
        
        if self.time == 0.:
        
            self.solve_steady_state_heat_driven_cavity()
            
            self.write_checkpoint(self.output_dir + "checkpoint_steady.h5")
            
            self.advance()
            
        self.setup_freezing_problem()
        
        if plot and (self.time == 0.):
        
            self.plot(savefigs = savefigs)
        
        table_filepath = self.output_dir + "DifferentiallyHeatedCavity_QoI_Table.txt"
        
        with open(table_filepath, "a") as results_file:

            results_file.write(
                "Ra_T, Ra_C, Sc, m_L, s, h, Delta_t, TimeOrder, t, A_S, M_C, A_phistar\n")
                
        def write_parameters():
        
            with open(table_filepath, "a") as results_file:
            
                results_file.write(
                    str(self.temperature_rayleigh_number.__float__()) + "," \
                    + str(self.concentration_rayleigh_number.__float__()) + ", " \
                    + str(self.schmidt_number.__float__()) + ", " \
                    + str(self.liquidus_slope.__float__()) + ", " \
                    + str(self.regularization_smoothing_parameter.__float__()) + ", " \
                    + str(1./float(self.uniform_gridsize)) + ", " \
                    + str(self.timestep_size.__float__()) + ", " \
                    + str(self.time_order) + ", " \
                    + str(self.time) + ", ")
                    
        def write_results():
        
            with open(table_filepath, "a") as results_file:
            
                solid_area = fenics.assemble(self.solid_area_integrand())
                
                solute_mass = fenics.assemble(self.solute_mass_integrand())
                
                area_above_critical_phi = fenics.assemble(self.area_above_critical_phi_integrand())
                
                results_file.write(str(solid_area) + ", " + str(solute_mass) + ", " + str(area_above_critical_phi))
        
        def write_newline():
        
            with open(table_filepath, "a") as results_file:
                
                results_file.write("\n")
        
        if self.time == 0.:
            
            write_parameters()
            
            write_results()
        
        time_tolerance = 1.e-8
        
        while (self.time < (endtime - time_tolerance)):
            
            write_newline()
            
            self._times[0] = self._times[1] + self.timestep_size
            
            write_parameters()
            
            self.solve_with_auto_regularization(
                enable_newton_solution_backup = True,
                max_attempts = max_regularization_attempts)
            
            write_results()
            
            if phaseflow.helpers.float_in(self.time, checkpoint_times):
                
                self.write_checkpoint(self.output_dir + "checkpoint_t" + str(self.time) + ".h5")
            
            if plot:
        
                self.plot(savefigs = savefigs)
            
            self.advance()
            
        write_newline()
        