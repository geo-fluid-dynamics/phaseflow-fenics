import phaseflow
import fenics


class HeatDrivenCavityBenchmarkSimulation(
        phaseflow.abstract_heated_cavity_phasechange_simulation.AbstractHeatedCavityPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            initial_uniform_gridsize = 8,
            setup_solver = True):
        
        self.initial_uniform_gridsize = initial_uniform_gridsize
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
        self.hot_wall_temperature.assign(0.5)
        
        self.cold_wall_temperature.assign(-0.5)
        
        self.timestep_size = 1.e-3
        
        self.temperature_rayleigh_number.assign(1.e6)
        
        self.prandtl_number.assign(0.71)
        
        """ Disable phase-change.
        Oddly enough, simply specifying an arbitrarily low melting temperature
        results in the Newton solver returning NaN's, e.g. the following did not work:
        
            self.regularization_central_temperature.assign(1.e-32)
            
        As a reminder of this for future development, let's assign something that works here.
        """
        self.pure_liquidus_temperature.assign(0.)
        
        self.regularization_central_temperature_offset.assign(-1.)
        
        self.solid_viscosity.assign(self.liquid_viscosity)
        
        self.stefan_number.assign(1.e32)
        
        """ Disable concentration equation """
        self.concentration_rayleigh_number.assign(0.)
        
        self.schmidt_number.assign(1.e32)
        
        self.liquidus_slope.assign(0.)
        
    def initial_mesh(self):
        
        return self.coarse_mesh()
        
    def initial_values(self):
        
        p0, u0_0, u0_1 = "0.", "0.", "0."
        
        T0 = "T_h + (T_c - T_h)*x[0]"
        
        C0 = "0."
        
        w0 = fenics.interpolate(
            fenics.Expression(
                (p0, u0_0, u0_1, T0, C0),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                element = self.element()),
            self.function_space)
        
        return w0
        
    def adaptive_goal(self):
        """ The default refinement algorithm has a 'known bug' which prevents us from 
        integrating a goal over a subdomain, referenced in the following:
        - https://fenicsproject.org/qa/6719/using-adapt-on-a-meshfunction-looking-for-a-working-example/
        - https://bitbucket.org/fenics-project/dolfin/issues/105/most-refinement-algorithms-do-not-set
        So we follow their advice and set the following algorithim.
        """
        fenics.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        
        return self.cold_wall_heat_flux_integrand()
        
        
def unsteadiness(sim):
        
    time_residual = \
        fenics.Function(sim.function_space)

    time_residual.assign(
        sim._solutions[0].leaf_node() - sim._solutions[1].leaf_node())

    L2_norm_relative_time_residual = fenics.norm(
        time_residual, "L2")/ \
        fenics.norm(sim._solutions[1].leaf_node(), "L2")
        
    return L2_norm_relative_time_residual
    
    
def test__heat_driven_cavity__ci__():

    verified_cold_wall_heat_flux = -8.9
    
    tolerance = 0.1
    
    sim = HeatDrivenCavityBenchmarkSimulation()
    
    sim.assign_initial_values()
    
    max_timesteps = 20
    
    steady = False
    
    for it in range(max_timesteps):

        sim.solve(goal_tolerance = tolerance)

        _unsteadiness = unsteadiness(sim)

        sim.advance()

        if (_unsteadiness <= 1.e-3):
            
            steady = True
            
            break
            
        sim.timestep_size *= 2.
        
    assert(steady)
    
    cold_wall_heat_flux = fenics.assemble(sim.cold_wall_heat_flux_integrand())
    
    print("Integrated cold wall heat flux = " + str(cold_wall_heat_flux))
    
    assert(abs(cold_wall_heat_flux - verified_cold_wall_heat_flux) < tolerance)
    

class WaterHeatDrivenCavityBenchmarkSimulation(
        phaseflow.abstract_heated_cavity_phasechange_simulation.AbstractHeatedCavityPhaseChangeSimulation):
    """ This class implements a benchmark simulation for the heat-driven cavity with water. """
    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            setup_solver = True):
        
        self.hot_wall_temperature_degC = fenics.Constant(
            10., name = "T_h_degC")
        
        self.cold_wall_temperature_degC = fenics.Constant(
            0., name = "T_c_degC")
        
        self.pure_liquidus_temperature_degC = fenics.Constant(
            0., name = "T_m_degC")
        
        self.pure_liquidus_temperature = self.T(
            self.pure_liquidus_temperature_degC)
        
        class HotWall(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[0], 0.)
                
        class ColdWall(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[0], 1.)
                
        class Walls(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary

        self._HotWall = HotWall
        
        self.hot_wall = self._HotWall()
       
        self._ColdWall = ColdWall
        
        self.cold_wall = self._ColdWall()
        
        self._Walls = Walls
        
        self.walls = self._Walls()
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
        self.temperature_rayleigh_number.assign(2.518084e6)
        
        self.prandtl_number.assign(6.99)
        
        """ Disable freezing. """
        self.pure_liquidus_temperature.assign(0.)
        
        self.regularization_central_temperature_offset.assign(0.)
        
        self.solid_viscosity.assign(self.liquid_viscosity)
        
        self.stefan_number.assign(1.e32)
        
        """ Disable concentration equation """
        self.concentration_rayleigh_number.assign(0.)
        
        self.schmidt_number.assign(1.e32)
        
        self.liquidus_slope.assign(0.)
        
    def T(self, T_degC):
        """ Normalize the temperature per the scaling of the governing equations. """
        T_h_degC = self.hot_wall_temperature_degC
    
        T_c_degC = self.cold_wall_temperature_degC
        
        T_m_degC = self.pure_liquidus_temperature_degC
        
        return (T_degC - T_m_degC)/(T_h_degC - T_c_degC)
        
    def T_degC(self, T):
    
        T_h_degC = self.hot_wall_temperature_degC
    
        T_c_degC = self.cold_wall_temperature_degC
        
        T_m_degC = self.pure_liquidus_temperature_degC

        return T*(T_h_degC - T_c_degC) + T_m_degC
        
    def solve(self, goal_tolerance = None):
        """ Validate parameters before solving. """
        assert(self.pure_liquidus_temperature.__float__() == self.T(self.pure_liquidus_temperature_degC).__float__())
        
        super().solve(goal_tolerance = goal_tolerance)
        
    def buoyancy(self, T, C):

        T_anomaly_degC = fenics.Constant(4.0293)  # [deg C]
        
        rho_anomaly_SI = fenics.Constant(999.972)  # [kg/m^3]
        
        w = fenics.Constant(9.2793e-6)  # [(deg C)^(-q)]
        
        q = fenics.Constant(1.894816)
        
        def rho_of_T_degC(T_degC):
            """ Eq. (24) from @cite{danaila2014newton} """
            return rho_anomaly_SI*(1. - w*abs(T_degC - T_anomaly_degC)**q)
            
        def rho(T):
            """ The normalized temperature is used by Eq. (25) from @cite{danaila2014newton} """
            return rho_of_T_degC(self.T_degC(T))
        
        beta = fenics.Constant(6.91e-5)  # [K^-1]
        
        T_hot_degC = self.hot_wall_temperature_degC
    
        T_cold_degC = self.cold_wall_temperature_degC
        
        Pr = self.prandtl_number
        
        Ra_T = self.temperature_rayleigh_number
        
        ghat = fenics.Constant((0., -1.), name = "ghat")
        
        T_m = self.pure_liquidus_temperature
        
        return Ra_T/Pr/(beta*(T_hot_degC - T_cold_degC))* \
            (rho(T_m) - rho(T))/rho(T_m)*ghat
            # Eq. (25) from @cite{danaila2014newton}
        
    def coarse_mesh(self):
        
        return fenics.UnitSquareMesh(2, 2)
        
    def initial_mesh(self):
        
        return self.coarse_mesh()
    
    def initial_values(self):
        
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", 
                 "0.", 
                 "0.", 
                 "(T_c - T_h)*x[0] + T_h", 
                 "0."),
                T_h = self.T(self.hot_wall_temperature_degC).__float__(), 
                T_c = self.T(self.cold_wall_temperature_degC).__float__(),
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
                self.T(self.hot_wall_temperature_degC).__float__(), 
                self.hot_wall),
            fenics.DirichletBC(
                self.function_space.sub(2), 
                self.T(self.cold_wall_temperature_degC).__float__(), 
                self.cold_wall)]
    
    def adaptive_goal(self):
        """ The default refinement algorithm has a 'known bug' which prevents us from 
        integrating a goal over a subdomain, referenced in the following:
        - https://fenicsproject.org/qa/6719/using-adapt-on-a-meshfunction-looking-for-a-working-example/
        - https://bitbucket.org/fenics-project/dolfin/issues/105/most-refinement-algorithms-do-not-set
        So we follow their advice and set the following algorithim.
        """
        fenics.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        
        return self.cold_wall_heat_flux_integrand()
        
    def deepcopy(self):
    
        sim = super().deepcopy()
        
        sim.hot_wall_temperature_degC.assign(self.hot_wall_temperature_degC)
        
        sim.cold_wall_temperature_degC.assign(self.cold_wall_temperature_degC)
        
        return sim
    
    
def test__heat_driven_cavity_water__ci__():

    verified_cold_wall_heat_flux = -7.5
    
    sim = WaterHeatDrivenCavityBenchmarkSimulation()

    sim.assign_initial_values()

    sim.timestep_size = 1.e-3

    sim.solver.parameters["newton_solver"]["maximum_iterations"] = 8

    sim.solve(goal_tolerance = 0.0001)
    
    for it in range(20):

        sim.advance()

        print("Solving time step number " + str(it))

        sim.timestep_size = 2.*sim.timestep_size

        for attempts in range(3):

            try:

                print("Trying time step size " + str(sim.timestep_size))

                sim.solve(goal_tolerance = 0.05)

                failed = False

                break

            except:

                sim.timestep_size = sim.timestep_size/2.

                sim.reset_initial_guess()

                failed = True

        if failed:

            break

        _unsteadiness = unsteadiness(sim)
        
        print("Unsteadiness = " + str(_unsteadiness))

        steady_tolerance = 0.01
        
        if _unsteadiness <= steady_tolerance:

            print("Reached steady state (tol = " + str(steady_tolerance) + ")")

            break

    cold_wall_heat_flux = fenics.assemble(sim.cold_wall_heat_flux_integrand())

    print("Integrated cold wall heat flux = " + str(cold_wall_heat_flux))
    
    assert(abs(cold_wall_heat_flux - verified_cold_wall_heat_flux) < 0.1)
    
    
class StefanProblemBenchmarkSimulation(
        phaseflow.abstract_heated_cavity_phasechange_simulation.AbstractHeatedCavityPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            setup_solver = True):
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
        self.timestep_size = 1.e-3
        
        self.temperature_rayleigh_number.assign(0.)
        
        self.prandtl_number.assign(1.)
        
        self.regularization_smoothing_parameter.assign(0.005)
        
    def buoyancy(self, T, C):
        """ While theoretically we can disable buoyancy via the Rayleigh number,
        the variational form requires the vector-valued term.
        The parent class is 2D, so we set an arbitrary 1D value here.
        """
        return fenics.Constant((0.,))
        
    def coarse_mesh(self):
        
        self.initial_uniform_cell_count = 4
        
        return fenics.UnitIntervalMesh(self.initial_uniform_cell_count)
        
    def initial_mesh(self):
    
        self.initial_hot_wall_refinement_cycles = 8
        
        mesh = self.coarse_mesh()
        
        for i in range(self.initial_hot_wall_refinement_cycles):
        
            cell_markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim(), False)
            
            cell_markers.set_all(False)
        
            for cell in fenics.cells(mesh):
                
                found_left_boundary = False
                
                for vertex in fenics.vertices(cell):
                    
                    if fenics.near(vertex.x(0), 0.):
                        
                        found_left_boundary = True
                        
                        break
                        
                if found_left_boundary:
                    
                    cell_markers[cell] = True
                    
                    break # There should only be one such point in 1D.
                    
            mesh = fenics.refine(mesh, cell_markers)
            
        return mesh
    
    def initial_values(self):
    
        initial_values = fenics.interpolate(
            fenics.Expression(("0.", "0.", "(T_h - T_c)*(x[0] < x_m0) + T_c", "0."),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                x_m0 = 1./self.initial_uniform_cell_count \
                    /2.**(self.initial_hot_wall_refinement_cycles - 1),
                element = self.element()),
            self.function_space)
            
        return initial_values
    
    def boundary_conditions(self):
    
        return [
            fenics.DirichletBC(
                self.function_space.sub(2), 
                self.hot_wall_temperature, 
                self.hot_wall),
            fenics.DirichletBC(
                self.function_space.sub(2), 
                self.cold_wall_temperature, 
                self.cold_wall)]
            
    def melted_length_integrand(self):
        
        p, u, T, C = fenics.split(self.solution)
        
        phi = self.semi_phasefield(T = T, C = C)
        
        dx = self.integration_measure
        
        return (1. - phi)*dx
        
    def adaptive_goal(self):
        
        return self.melted_length_integrand()
        
        
def test__stefan_problem__ci__():

    expected_melted_length = 0.094662
    
    tolerance = 1.e-6

    sim = StefanProblemBenchmarkSimulation()
    
    sim.assign_initial_values()
    
    """ Set a zero initial guess for the Newton method.
    
    By default, a Simulation uses the latest solution (or the initial values) as the initial guess.
    For the special case of the 1D Stefan problem, that approach fails for yet unknown reasons.
    """
    sim._solutions[0].vector()[:] = 0.
    
    end_time = 0.1
    
    timestep_count = 100
    
    sim.timestep_size = end_time/float(timestep_count)
    
    for it in range(timestep_count):
        
        sim.solve(goal_tolerance = tolerance)
        
        sim.advance()
    
    melted_length = fenics.assemble(sim.melted_length_integrand())
    
    assert(abs(melted_length - expected_melted_length) < tolerance)


def test__stefan_problem_with_bdf2__regression__ci__():

    expected_melted_length = 0.094662
    
    sim = StefanProblemBenchmarkSimulation(time_order = 2)
    
    sim.assign_initial_values()
    
    end_time = 0.1
    
    timestep_count = 25
    
    sim.timestep_size = end_time/float(timestep_count)
    
    sim.regularization_smoothing_parameter.assign(0.005)
    
    for it in range(timestep_count):
    
        if it == 1:
        
            sim.regularization_sequence = None
            
        sim.solve_with_auto_regularization(goal_tolerance = 1.e-7)
        
        sim.advance()
    
    melted_length = fenics.assemble(sim.melted_length_integrand())
    
    tolerance = 1.e-4
    
    assert(abs(melted_length - expected_melted_length) < tolerance)