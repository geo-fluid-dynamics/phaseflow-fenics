import phaseflow
import fenics

 
class Benchmark:
 
    def __init__(self):
    
        self.model = None
        
        self.solver = None
        
        self.timestepper = None
        
        self.adaptive_goal_integrand = None
        
        self.adaptive_solver_tolerance = 1.e-4
        
        self.output_dir = "output/benchmarks/"
        
        self.end_time = None
        
        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = True
        
        self.adaptive_time_power = 1.
        
        self.nlp_max_iterations = 50
        
        self.nlp_relaxation = 1.
        
        self.nlp_absolute_tolerance = 1.e-10
        
        self.nlp_relative_tolerance = 1.e-9
        
        
    def verify(self):
        
        assert(False)
        
    
    def verify_scalar_solution_component(self, component, points, verified_values, 
        relative_tolerance, absolute_tolerance):
    
        assert(len(verified_values) == len(points))
        
        for i, verified_value in enumerate(verified_values):
        
            p = phaseflow.core.Point(points[i])
            
            if self.model.mesh.bounding_box_tree().collides_entity(p):
            
                values = self.model.state.solution.leaf_node()(p)
                
                value = values[component]
                
                absolute_error = abs(value - verified_value)
                
                if abs(verified_value) > absolute_tolerance:
                
                    relative_error = absolute_error/verified_value
               
                    assert(relative_error < relative_tolerance)
                    
                else:
                
                    assert(absolute_error < absolute_tolerance)
    
    
    def run(self):
    
        self.solver = phaseflow.core.Solver(
            model = self.model, 
            adaptive_goal_integrand = self.adaptive_goal_integrand, 
            adaptive_solver_tolerance = self.adaptive_solver_tolerance,
            nlp_max_iterations = self.nlp_max_iterations,
            nlp_relaxation = self.nlp_relaxation,
            nlp_absolute_tolerance = self.nlp_absolute_tolerance,
            nlp_relative_tolerance = self.nlp_relative_tolerance)
        
        self.timestepper = phaseflow.core.TimeStepper(
            model = self.model,
            solver = self.solver,
            output_dir = self.output_dir,
            end_time = self.end_time,
            stop_when_steady = self.stop_when_steady,
            steady_relative_tolerance = self.steady_relative_tolerance,
            adapt_timestep_to_unsteadiness = self.adapt_timestep_to_unsteadiness,
            adaptive_time_power = self.adaptive_time_power)
        
        self.timestepper.run_until_end_time()
            
        self.verify()
        
    
class Cavity(Benchmark):

    def __init__(self, mesh_size = [20, 20], xmin = 0., ymin = 0., xmax = 1., ymax = 1.):
    
        Benchmark.__init__(self)
        
        if type(mesh_size) is type(20):
        
            mesh_size = [mesh_size, mesh_size]
        
        if [xmin, ymin, xmax, ymax] == [0., 0., 1., 1.]:
        
            self.mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), 
                mesh_size[0], mesh_size[1], "crossed")
            
        else:
        
            self.mesh = fenics.RectangleMesh(fenics.mpi_comm_world(), 
                fenics.Point(xmin, ymin), fenics.Point(xmax, ymax),
                mesh_size[0], mesh_size[1], "crossed")
        
        
        self.left_wall = "near(x[0],  xmin)".replace("xmin", str(xmin))
        
        self.right_wall = "near(x[0],  xmax)".replace("xmax", str(xmax))
        
        self.bottom_wall = "near(x[1],  ymin)".replace("ymin",  str(ymin))
        
        self.top_wall = "near(x[1],  ymax)".replace("ymax", str(ymax))
        
        self.walls = \
            self.top_wall + " | " + self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
    
  
class LidDrivenCavity(Cavity):

    def __init__(self, 
            mesh_size = 20, 
            ymin = 0., 
            timestep_size = 1.e12,
            relative_tolerance = 3.e-2,
            absolute_tolerance = 1.e-2):
    
        Cavity.__init__(self, mesh_size = mesh_size, ymin = ymin)
        
        self.relative_tolerance = relative_tolerance
        
        self.absolute_tolerance = absolute_tolerance
        
        self.fixed_walls = self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        self.timestep_size = timestep_size
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = ("0.", self.top_wall, "0.", "1."),
            boundary_conditions = [
                {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
                {"subspace": 1, "location": self.fixed_walls, "value": (0., 0.)}],
            timestep_bounds = timestep_size,
            liquid_viscosity = 0.01)
        
        self.end_time = timestep_size
        
        self.output_dir = "output/benchmarks/lid_driven_cavity/"
    
    
    def verify(self):
        """ Verify against \cite{ghia1982}. """
        self.verify_scalar_solution_component(
            component = 1,
            points = [((self.xmin + self.xmax)/2., y) 
                for y in [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]],
            verified_values = [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, 
                -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000],
            relative_tolerance = self.relative_tolerance,
            absolute_tolerance = self.absolute_tolerance)
    
    
class AdaptiveLidDrivenCavity(LidDrivenCavity):

    def __init__(self, mesh_size = 2, ymin = 0., timestep_size = 1.e12):
    
        LidDrivenCavity.__init__(self, mesh_size = mesh_size, ymin = ymin, timestep_size = timestep_size)
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_integrand = u[0]*u[0]
        
        self.adaptive_solver_tolerance = 1.e-4
        
        self.output_dir = "output/benchmarks/adaptive_lid_driven_cavity/"

        
class LidDrivenCavityWithSolidSubdomain(LidDrivenCavity):

    y_pci = 0.
    
    def __init__(self, mesh_size = [20, 25], pci_refinement_cycles = 4, timestep_size = 20.):
    
        LidDrivenCavity.__init__(self, mesh_size = mesh_size, ymin = -0.25, timestep_size = timestep_size)
        
        self.relative_tolerance = 8.e-2  # @todo This is quite large.
        
        self.absolute_tolerance = 2.e-2
        
        self.refine_near_y_pci(pci_refinement_cycles = pci_refinement_cycles)
        
        self.setup_model(timestep_size = timestep_size)
        
        
    def refine_near_y_pci(self, pci_refinement_cycles = 4):   
        
        class PhaseInterface(fenics.SubDomain):
            
            def inside(self, x, on_boundary):
            
                return fenics.near(x[1], LidDrivenCavityWithSolidSubdomain.y_pci)
        
        
        phase_interface = PhaseInterface()
        
        for i in range(pci_refinement_cycles):
            
            edge_markers = fenics.EdgeFunction("bool", self.mesh)
            
            phase_interface.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
            
            self.mesh = self.mesh.child()
        
        
    def setup_model(self, timestep_size = 20.):
    
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = (
                "0.", 
                self.top_wall, 
                "0.", 
                "1. - 2.*(x[1] <= y_pci)".replace("y_pci", str(LidDrivenCavityWithSolidSubdomain.y_pci))),
            boundary_conditions = [
                {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
                {"subspace": 1, "location": self.fixed_walls, "value": (0., 0.)}],
            prandtl_number = 1.e16,
            liquid_viscosity = 0.01,
            solid_viscosity = 1.e6,
            stefan_number = 1.e16,
            semi_phasefield_mapping = phaseflow.pure.TanhSemiPhasefieldMapping(
                regularization_central_temperature = -0.01,
                regularization_smoothing_parameter = 0.01),
            timestep_bounds = timestep_size,
            quadrature_degree = 3)
        
        self.output_dir = "output/benchmarks/lid_driven_cavity_with_solid_subdomain/"
        
        
class AdaptiveLidDrivenCavityWithSolidSubdomain(LidDrivenCavityWithSolidSubdomain):
    """ Ideally we should be able to use AMR instead of manually refining the prescribed PCI.
    Unfortunately, the adaptive solver computes an error estimate of exactly 0,
    which seems to be a bug in FEniCS.
    We'll want to make a MWE and investigate. For now let's leave this failing benchmark here."""
    def __init__(self):
    
        LidDrivenCavityWithSolidSubdomain.__init__(self, 
            mesh_size = (4, 5),
            pci_refinement_cycles = 4)
        
        self.end_time = 2.*self.timestep_size
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_integrand = u[0]*u[0]
        
        self.adaptive_solver_tolerance = 1.e-5
        
        self.output_dir = "output/benchmarks/adaptive_lid_driven_cavity_with_solid_subdomain/"
    
    
class HeatDrivenCavity(Cavity):

    def __init__(self, mesh_size = 20):
    
        Cavity.__init__(self, mesh_size = mesh_size)
        
        T_hot = 0.5
    
        T_cold = -T_hot
        
        self.Ra = 1.e6
        
        self.Pr = 0.71
    
        initial_values = ("0.", "0.", "0.",
            "T_hot + x[0]*(T_cold - T_hot)".replace("T_hot", str(T_hot)).replace("T_cold", str(T_cold)))
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = initial_values,
            boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": T_hot},
                {"subspace": 2, "location": self.right_wall, "value": T_cold}],
            prandtl_number = self.Pr,
            buoyancy = phaseflow.pure.IdealizedLinearBoussinesqBuoyancy(
                rayleigh_numer = self.Ra, 
                prandtl_number = self.Pr),
            timestep_bounds = (1.e-4, 1.e-3, 1.e12))
            
        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = True
        
        self.output_dir = "output/benchmarks/heat_driven_cavity/"
        
        
    def verify(self):
        """ Verify against the result published in \cite{wang2010comprehensive}. """
        self.verify_scalar_solution_component(
            component = 1,
            points = [((self.xmin + self.xmax)/2., y) for y in [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.]],
            verified_values = [val*self.Ra**0.5/self.Pr 
                for val in [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]],
            relative_tolerance = 2.e-2,
            absolute_tolerance = 1.e-2*0.0649*self.Ra**0.5/self.Pr)
    
    
class AdaptiveHeatDrivenCavity(HeatDrivenCavity):
    
    def __init__(self, mesh_size = 2):
    
        HeatDrivenCavity.__init__(self, mesh_size = mesh_size)
        
        self.output_dir = "output/benchmarks/adaptive_heat_driven_cavity/"
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_integrand = u[0]*T
        
        self.adaptive_solver_tolerance = 1.e-2
        
        
class HeatDrivenCavityWithWater(Cavity):

    def __init__(self, mesh_size = 40):
    
        Cavity.__init__(self, mesh_size = mesh_size)
        
        self.Ra = 2.52e6
        
        self.Pr = 6.99
        
        T_hot = 10.  # [deg C]
    
        T_cold = 0.  # [deg C]
        
        scaled_T_hot = 1.
        
        scaled_T_cold = 0.
        
        initial_temperature = "scaled_T_hot + x[0]*(scaled_T_cold - scaled_T_hot)"
        
        initial_temperature = initial_temperature.replace("scaled_T_hot", str(scaled_T_hot))
        
        initial_temperature = initial_temperature.replace("scaled_T_cold", str(scaled_T_cold))
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = ("0.", "0.", "0.", initial_temperature),
            boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": scaled_T_hot},
                {"subspace": 2, "location": self.right_wall, "value": scaled_T_cold}],
            prandtl_number = self.Pr,
            buoyancy = phaseflow.pure.GebhartWaterBuoyancy(
                hot_temperature = T_hot,
                cold_temperature = T_cold,
                rayleigh_numer = self.Ra, 
                prandtl_number = self.Pr),
            timestep_bounds = (1.e-4, 1.e-3, 1.e-2))
            
        self.output_dir = "output/benchmarks/heat_driven_cavity_with_water/"

        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = True
        
        self.adaptive_time_power = 0.5
        
        self.nlp_max_iterations = 12
        
        
    def verify(self):
        """Verify against steady-state solution from michalek2003."""
        self.verify_scalar_solution_component(
            component = 3,
            points = [(x, (self.ymin + self.ymax)/2.) 
                for x in [0.00, 0.05, 0.12, 0.23, 0.40, 0.59, 0.80, 0.88, 1.00]],
            verified_values = [1.00, 0.66, 0.56, 0.58, 0.59, 0.62, 0.20, 0.22, 0.00],
            relative_tolerance = 5.e-2,
            absolute_tolerance = 1.e-2)
            
            
class AdaptiveHeatDrivenCavityWithWater(HeatDrivenCavityWithWater):
    
    def __init__(self, mesh_size = 4):
    
        HeatDrivenCavityWithWater.__init__(self, mesh_size = mesh_size)
        
        self.output_dir = "output/benchmarks/adaptive_heat_driven_cavity_with_water/"
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_integrand = u[0]*T
        
        self.adaptive_solver_tolerance = 1.e-2
        
        
    def run(self):
    
        Benchmark.run(self)
            
            
class StefanProblem(Benchmark):

    def refine_near_left_boundary(mesh, cycles):
        """ Refine mesh near the left boundary.
        The usual approach of using SubDomain and EdgeFunction isn't appearing to work
        in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
        """
        for i in range(cycles):
            
            cell_markers = fenics.CellFunction("bool", mesh)
            
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
            

    def __init__(self,
            T_hot = 1.,
            T_cold = -0.01,
            regularization_smoothing_parameter = 0.005,
            timestep_size = 1.e-3,
            initial_uniform_cell_count = 311,
            initial_hot_wall_refinement_cycles = 0,
            end_time = 0.1,
            quadrature_degree = None):
    
        Benchmark.__init__(self)
        
        initial_pci_position = \
            1./float(initial_uniform_cell_count)/2.**(initial_hot_wall_refinement_cycles - 1)
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(T_hot))
        
        initial_temperature = initial_temperature.replace("T_cold", str(T_cold))
        
        mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)

        mesh = StefanProblem.refine_near_left_boundary(mesh, initial_hot_wall_refinement_cycles)
        
        self.model = phaseflow.pure_isotropic.Model(
            mesh = mesh,
            initial_values = ("0.", "0.", initial_temperature),
            boundary_conditions = [
                {"subspace": 2, "location": "near(x[0],  0.)", "value": T_hot},
                {"subspace": 2, "location": "near(x[0],  1.)", "value": T_cold}],
            stefan_number = 0.045,
            semi_phasefield_mapping = phaseflow.pure.TanhSemiPhasefieldMapping(
                regularization_central_temperature = 0.,
                regularization_smoothing_parameter = regularization_smoothing_parameter),
            timestep_bounds = timestep_size,
            quadrature_degree = quadrature_degree)

        self.end_time = end_time
        
        self.output_dir = "output/benchmarks/stefan_problem/"
    
        self.stop_when_steady = False
        
    
    def verify(self):
        """ Verify against analytical solution. """
        self.verify_scalar_solution_component(
            component = 2,
            points = [0.00, 0.025, 0.050, 0.075, 0.10, 0.5, 1.],
            verified_values = [1.0, 0.73, 0.47, 0.20, 0.00, -0.01, -0.01],
            relative_tolerance = 2.e-2,
            absolute_tolerance = 1.e-2)
           

class AdaptiveStefanProblem(StefanProblem):

    def __init__(self):
    
        StefanProblem.__init__(self,
            initial_uniform_cell_count = 4,
            initial_hot_wall_refinement_cycles = 8)
        
        p, u, T = fenics.split(self.model.state.solution)
        
        phi = self.model.semi_phasefield_mapping.function
        
        self.adaptive_goal_integrand = phi(T)
        
        self.adaptive_solver_tolerance = 1.e-6
        
        self.output_dir = "output/benchmarks/adaptive_stefan_problem/"
        
        
    def run(self):
        """ This test fails with the usual initial guess of w^0 = w_n,
            but passes with w^0 = 0.
        """
        self.solver = phaseflow.core.Solver(
            model = self.model, 
            initial_guess = ("0.", "0.", "0."),
            adaptive_goal_integrand = self.adaptive_goal_integrand, 
            adaptive_solver_tolerance = self.adaptive_solver_tolerance)
        
        self.timestepper = phaseflow.core.TimeStepper(
            model = self.model,
            solver = self.solver,
            output_dir = self.output_dir,
            end_time = self.end_time,
            stop_when_steady = self.stop_when_steady,
            steady_relative_tolerance = self.steady_relative_tolerance,
            adapt_timestep_to_unsteadiness = self.adapt_timestep_to_unsteadiness,
            adaptive_time_power = self.adaptive_time_power)
        
        self.timestepper.run_until_end_time()
            
        self.verify()
        
            
class AdaptiveConvectionCoupledMeltingOctadecanePCM(Cavity):

    def __init__(self, 
            T_hot = 1.,
            T_cold = -0.01,
            stefan_number = 0.045,
            rayleigh_number = 3.27e5,
            prandtl_number = 56.2,
            solid_viscosity = 1.e8,
            liquid_viscosity = 1.,
            timestep_size = 1.,
            initial_mesh_size = 1,
            initial_hot_wall_refinement_cycles = 6,
            initial_pci_position = None,
            regularization_central_temperature = 0.01,
            regularization_smoothing_parameter = 0.025,
            end_time = 80.,
            adaptive_solver_tolerance = 1.e-5,
            quadrature_degree = 8,
            nlp_max_iterations = 100):
    
        Cavity.__init__(self, mesh_size = initial_mesh_size)
        
        
        # Make the mesh with initial refinements near the hot wall.
        class HotWall(fenics.SubDomain):
        
            def inside(self, x, on_boundary):
            
                return on_boundary and fenics.near(x[0], 0.)

            
        hot_wall = HotWall()
        
        for i in range(initial_hot_wall_refinement_cycles):
            
            edge_markers = fenics.EdgeFunction("bool", self.mesh)
            
            hot_wall.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
            
            self.mesh = self.mesh.child()

            
        # Set up the model.
        if initial_pci_position == None:
        
            initial_pci_position = \
                1./float(initial_mesh_size)/2.**(initial_hot_wall_refinement_cycles - 1)
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(T_hot))
        
        initial_temperature = initial_temperature.replace("T_cold", str(T_cold))
        
        self.model = phaseflow.pure_isotropic.Model(
            mesh = self.mesh,
            initial_values = ("0.", "0.", "0.", initial_temperature),
            boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": T_hot},
                {"subspace": 2, "location": self.right_wall, "value": T_cold}],
            stefan_number = stefan_number,
            prandtl_number = prandtl_number,
            buoyancy = phaseflow.pure.IdealizedLinearBoussinesqBuoyancy(
                rayleigh_numer = rayleigh_number, 
                prandtl_number = prandtl_number),
            semi_phasefield_mapping = phaseflow.pure.TanhSemiPhasefieldMapping(
                regularization_central_temperature = regularization_central_temperature,
                regularization_smoothing_parameter = regularization_smoothing_parameter),
            timestep_bounds = timestep_size,
            quadrature_degree = quadrature_degree)
        
        phi = self.model.semi_phasefield_mapping.function
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_integrand = phi(T)
        
        self.adaptive_solver_tolerance = adaptive_solver_tolerance
        
        self.end_time = end_time
        
        self.output_dir = "output/benchmarks/adaptive_convection_coupled_melting_octadecane_pcm/"
            
            

        
            
if __name__=='__main__':

    LidDrivenCavity().run()
    
    AdaptiveLidDrivenCavity().run()
    
    LidDrivenCavityWithSolidSubdomain().run()
    
    HeatDrivenCavity().run()
    
    AdaptiveHeatDrivenCavity.run()
    
    HeatDrivenCavityWithWater().run()
    
    AdaptiveHeatDrivenCavityWithWater().run()
    
    StefanProblem.run()
    
    AdaptiveStefanProblem.run()
    
    AdaptiveConvectionCoupledMeltingOctadecanePCM.run()
    