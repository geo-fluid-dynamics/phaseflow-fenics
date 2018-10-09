import phaseflow
import fenics


class CavityMeltingAndRefreezingCycleSimulation(
        phaseflow.abstract_phasechange_simulation.AbstractPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            quadrature_degree = 3,
            uniform_gridsize = 20,
            setup_solver = True):
        
        self.uniform_gridsize = uniform_gridsize
        
        
        self.hot_wall_temperature = fenics.Constant(1., name = "T_h")
        
        self.cold_wall_temperature = fenics.Constant(-1., name = "T_c")
        
        self.initial_concentration = fenics.Constant(1., name = "C0")
        
        
        self.hot_wall_temperature_during_melting = fenics.Constant(1.)
        
        self.cold_wall_temperature_during_melting = fenics.Constant(0.)
        
        
        self.hot_wall_temperature_during_freezing = fenics.Constant(0.)
        
        self.cold_wall_temperature_during_freezing = fenics.Constant(-1.)
        
        
        if quadrature_degree is None:
        
            integration_measure = fenics.dx
            
        else:
        
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8})
        
        self.hot_wall_temperature = fenics.Constant(1., name = "T_h")
        
        self.cold_wall_temperature = fenics.Constant(-0.01, name = "T_c")
        
        self.initial_concentration = fenics.Constant(1., name = "C0")
   
        class HotWall(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[1], 0.)
                
        class ColdWall(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[1], 1.)
                
        class Walls(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary

        self._HotWall = HotWall
        
        self.hot_wall = self._HotWall()
       
        self._ColdWall = ColdWall
        
        self.cold_wall = self._ColdWall()
        
        self._Walls = Walls
        
        self.walls = self._Walls()
        
        self.uniform_gridsize = uniform_gridsize
        
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
        
        self.hot_wall_temperature.assign(self.hot_wall_temperature_during_melting)
        
        self.cold_wall_temperature.assign(self.cold_wall_temperature_during_melting)
        
        
        if setup_solver:
        
            self.solver.parameters["newton_solver"]["maximum_iterations"] = 12
        
            self.max_regularization_attempts = 16
            
        self.results_table_filepath = self.output_dir + "ResultsTable.txt"
        
        self.checkpoint_times = (0., 1., 2., 4., 8., 16., 32., 64., 128.)
        
    def coarse_mesh(self):
        
        M = self.uniform_gridsize
        
        return fenics.UnitSquareMesh(M, M)
    
    def initial_values(self):
        
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", 
                 "0.", 
                 "0.", 
                 "(T_h - T_c)*(x[1] < y_m0) + T_c",
                 "C_0"),
                y_m0 = 0.1,
                T_h = self.hot_wall_temperature_during_melting, 
                T_c = self.cold_wall_temperature_during_melting,
                C_0 = self.initial_concentration,
                element = self.element()),
            self.function_space)
            
        return initial_values
    
    def boundary_conditions(self):
    
        return [
            fenics.DirichletBC(
                self.function_space.sub(1), 
                (0., 0.), 
                self.walls),
            fenics.DirichletBC(
                self.function_space.sub(2), 
                self.hot_wall_temperature, 
                self.hot_wall),
            fenics.DirichletBC(
                self.function_space.sub(2), 
                self.cold_wall_temperature, 
                self.cold_wall)]
    
    def setup_melting_problem(self):
    
        self.cold_wall_temperature.assign(self.cold_wall_temperature_during_melting)
        
        self.hot_wall_temperature.assign(self.hot_wall_temperature_during_melting)
        
    def setup_freezing_problem(self):
        
        self.cold_wall_temperature.assign(self.cold_wall_temperature_during_freezing)
        
        self.hot_wall_temperature.assign(self.hot_wall_temperature_during_freezing)
        
    def write_results_table_header(self, table_filepath):
        
        with open(table_filepath, "a") as table_file:

            table_file.write(
                "gamma, mu_S, Ra_T, Ra_C, Pr, Ste, Sc, m_L" + \
                ", T_m, delta_T, s, h, Delta_t, TimeOrder, QuadratureDegree" + \
                ", t" + \
                ", A_S, A_phistar, M_C" + \
                ", p_min, p_max, u_Linf_norm, T_min, T_max, C_min, C_max" + \
                ", phi_min, phi_max" + \
                "\n")
                
    def write_results_table_row(self, table_filepath):
    
        with open(table_filepath, "a") as table_file:
        
            table_file.write(
                str(self.pressure_penalty_factor.__float__()) + "," \
                + str(self.solid_viscosity.__float__()) + "," \
                + str(self.temperature_rayleigh_number.__float__()) + "," \
                + str(self.concentration_rayleigh_number.__float__()) + ", " \
                + str(self.prandtl_number.__float__()) + ", " \
                + str(self.stefan_number.__float__()) + ", " \
                + str(self.schmidt_number.__float__()) + ", " \
                + str(self.liquidus_slope.__float__()) + ", " \
                + str(self.pure_liquidus_temperature.__float__()) + ", " \
                + str(self.regularization_central_temperature_offset.__float__()) + ", " \
                + str(self.regularization_smoothing_parameter.__float__()) + ", " \
                + str(1./float(self.uniform_gridsize)) + ", " \
                + str(self.timestep_size.__float__()) + ", " \
                + str(self.time_order) + ", " \
                + str(self.integration_measure.metadata()["quadrature_degree"]) + ", " \
                + str(self.time) + ", ")
                
            solid_area = fenics.assemble(self.solid_area_integrand())
            
            area_above_critical_phi = fenics.assemble(
                self.area_above_critical_phi_integrand())
            
            solute_mass = fenics.assemble(self.solute_mass_integrand())
            
            p, u, T, C = self.solution.leaf_node().split(deepcopy = True)
            
            phi = fenics.project(
                self.semi_phasefield(T = T, C = C), mesh = self.mesh.leaf_node())
    
            Cbar = fenics.project(C*(1. - phi), mesh = self.mesh.leaf_node())
    
            table_file.write(
                str(solid_area) + ", " \
                + str(area_above_critical_phi) + ", " \
                + str(solute_mass) + ", " \
                + str(p.vector().min()) + ", " \
                + str(p.vector().max()) + ", " \
                + str(fenics.norm(u.vector(), "linf"))  + ", " \
                + str(T.vector().min()) + ", " \
                + str(T.vector().max()) + ", " \
                + str(Cbar.vector().min()) + ", " \
                + str(Cbar.vector().max()) + ", " \
                + str(phi.vector().min()) + ", " \
                + str(phi.vector().max()))
            
            table_file.write("\n")
        
    def run_until(self, endtime = 1., plot = False, savefigs = False):
    
        time_tolerance = 1.e-8
        
        while (self.time < (endtime - time_tolerance)):
                
            self.solve_with_auto_regularization(
                enable_newton_solution_backup = True,
                max_attempts = self.max_regularization_attempts)
            
            self.write_results_table_row(self.results_table_filepath)
            
            with open(self.output_dir + "regularization_history.txt", "a") \
                    as regularization_history_file:
                
                regularization_history_file.write(str(self.regularization_sequence))
            
            if phaseflow.helpers.float_in(self.time, self.checkpoint_times):
                
                self.write_checkpoint(self.output_dir + "checkpoint_t" + str(self.time) + ".h5")
            
            if plot:
        
                self.plot(savefigs = savefigs)
            
            self.advance()
    
    def run(self,
            melting_endtime = 1., 
            freezing_endtime = 2.,
            checkpoint_times = (0., 1., 2., 4., 8., 16., 32., 64., 128.),            
            max_regularization_attempts = 16, 
            plot = False, 
            savefigs = False):
        
        phaseflow.helpers.mkdir_p(self.output_dir)
        
        assert(self.temperature_rayleigh_number.__float__() > 1.e-8)
        
        self.max_regularization_attempts = max_regularization_attempts
        
        self.checkpoint_times = checkpoint_times
        
        self.setup_melting_problem()
        
        self.assign_initial_values()
        
        self.results_table_filepath = self.output_dir + "ResultsTable.txt"
        
        print("Writing results table to " + str(self.results_table_filepath))
        
        self.write_nonlinear_solver_table_header()
        
        self.write_results_table_header(self.results_table_filepath)
        
        self.write_results_table_row(self.results_table_filepath)
        
        if plot:
    
            self.plot(savefigs = savefigs)
        
        self.run_until(endtime = melting_endtime, plot = plot, savefigs = savefigs)
        
        self.setup_freezing_problem()
        
        self.run_until(endtime = freezing_endtime, plot = plot, savefigs = savefigs)
     