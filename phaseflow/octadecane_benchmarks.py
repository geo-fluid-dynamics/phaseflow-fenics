"""**benchmarks.py** applies Phaseflow to a variety of benchmark problems."""
import phaseflow
import phaseflow.octadecane
import fenics

 
class BenchmarkSimulation(phaseflow.octadecane.Simulation):
 
    def __init__(self):
    
        phaseflow.octadecane.Simulation.__init__(self)
        
        self.output_dir += "benchmark/"

    
    def run(self):
        
        phaseflow.simulation.Simulation.run(self)
        
        self.verify()
        
        
    def verify(self):
        """ This must be overloaded. """
        assert(False)
        
    
    def verify_scalar_solution_component(self, 
            component, 
            points, 
            verified_values, 
            relative_tolerance, 
            absolute_tolerance):
    
        assert(len(verified_values) == len(points))
        
        for i, verified_value in enumerate(verified_values):
        
            p = phaseflow.helpers.Point(points[i])
            
            if self.mesh.bounding_box_tree().collides_entity(p):
            
                values = self.state.solution.leaf_node()(p)
                
                value = values[component]
                
                absolute_error = abs(value - verified_value)
                
                if abs(verified_value) > absolute_tolerance:
                
                    relative_error = absolute_error/verified_value
               
                    assert(relative_error < relative_tolerance)
                    
                else:
                
                    assert(absolute_error < absolute_tolerance)
        
    
class CavityBenchmarkSimulation(BenchmarkSimulation):

    def __init__(self, mesh_size = (20, 20), 
            xmin = 0., ymin = 0., zmin = None, xmax = 1., ymax = 1., zmax = None):
    
        BenchmarkSimulation.__init__(self)
        
        if type(mesh_size) is type(20):
        
            mesh_size = [mesh_size, mesh_size]
    
        if len(mesh_size) == 2:
    
            if (xmin, ymin, xmax, ymax) == (0., 0., 1., 1.):
    
                self.mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), 
                    mesh_size[0], mesh_size[1], "crossed")
        
            else:
            
                self.mesh = fenics.RectangleMesh(fenics.mpi_comm_world(), 
                    fenics.Point(xmin, ymin), fenics.Point(xmax, ymax),
                    mesh_size[0], mesh_size[1], "crossed")
    
        elif len(mesh_size) == 3:
        
            self.mesh = fenics.BoxMesh(fenics.mpi_comm_world(), 
                fenics.Point(xmin, ymin, zmin), fenics.Point(xmax, ymax, zmax),
                mesh_size[0], mesh_size[1], mesh_size[2])
        
        self.left_wall = "near(x[0],  xmin)".replace("xmin", str(xmin))
        
        self.right_wall = "near(x[0],  xmax)".replace("xmax", str(xmax))
        
        self.bottom_wall = "near(x[1],  ymin)".replace("ymin",  str(ymin))
        
        self.top_wall = "near(x[1],  ymax)".replace("ymax", str(ymax))
        
        self.walls = \
                self.top_wall + " | " + self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        if len(mesh_size) == 3:
        
            self.back_wall = "near(x[2],  zmin)".replace("zmin",  str(zmin))
        
            self.front_wall = "near(x[2],  zmax)".replace("zmax", str(zmax))
        
            self.walls += " | " + self.back_wall + " | " + self.front_wall
            
        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = \
            xmin, ymin, zmin, xmax, ymax, zmax
    
  
class LidDrivenCavityBenchmarkSimulation(CavityBenchmarkSimulation):

    def __init__(self, 
            mesh_size = 2, 
            ymin = 0., 
            timestep_size = 1.e12):
    
        CavityBenchmarkSimulation.__init__(self, mesh_size = mesh_size, ymin = ymin)
        
        self.relative_tolerance = 3.e-2
        
        self.absolute_tolerance = 1.e-2
        
        self.fixed_walls = self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        self.timestep_size = timestep_size
        
        self.boundary_conditions = [
            {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
            {"subspace": 1, "location": self.fixed_walls, "value": (0., 0.)}]
            
        self.timestep_size = timestep_size
        
        self.liquid_viscosity = 0.01
        
        self.end_time = timestep_size
        
        self.output_dir += "lid_driven_cavity/"
        
        self.adaptive_goal_tolerance = 1.e-4
    
    
    def update_initial_values(self):
    
        self.old_state.interpolate(("0.", self.top_wall, "0.", "1."))
    
    
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.state.solution)
        
        phi = self.semi_phasefield_mapping
        
        self.adaptive_goal_form = u[0]*u[0]*self.integration_metric
        
    
    def verify(self):
        """ Verify against \cite{ghia1982}. """
        self.verify_scalar_solution_component(
            component = 1,
            points = [(0.5, y) 
                for y in [1.0000, 0.9766, 0.1016, 0.0547, 0.0000]],
            verified_values = [1.0000, 0.8412, -0.0643, -0.0372, 0.0000],
            relative_tolerance = self.relative_tolerance,
            absolute_tolerance = self.absolute_tolerance)
    
        
class LDCBenchmarkSimulationWithSolidSubdomain(LidDrivenCavityBenchmarkSimulation):
    """ Similar to the lid-driven cavity, but extended with a solid subdomain to test variable viscosity.
    
    The unit square from the original benchmark is prescribed a temperature which makes it fluid,
    while below the original bottom wall, a cold temperature is prescribed, making it solid.
    """
    y_pci = 0.
    
    def __init__(self, 
            mesh_size = [4, 5], 
            pci_refinement_cycles = 4, 
            timestep_size = 20.):
    
        LidDrivenCavityBenchmarkSimulation.__init__(self, 
            mesh_size = mesh_size, ymin = -0.25, timestep_size = timestep_size)
        
        self.relative_tolerance = 8.e-2  # @todo This is quite large.
        
        self.absolute_tolerance = 2.e-2
        
        self.refine_near_y_pci(pci_refinement_cycles = pci_refinement_cycles)
        
        self.adaptive_goal_tolerance = 1.e-5
        
        self.boundary_conditions = [
                {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
                {"subspace": 1, "location": self.fixed_walls, "value": (0., 0.)}]
                
        self.prandtl_number = 1.e16
        
        self.liquid_viscosity = 0.01
        
        self.solid_viscosity = 1.e6
        
        self.stefan_number = 1.e16
        
        self.regularization_central_temperature = -0.01
        
        self.regularization_smoothing_parameter = 0.01
        
        self.timestep_size = timestep_size
        
        self.end_time = 2.*timestep_size
        
        self.quadrature_degree = 3
        
        self.output_dir += "with_solid_subdomain/"
        
    
    def update_initial_values(self):
    
        temperature_string = "1. - 2.*(x[1] <= y_pci)".replace("y_pci", 
                str(LDCBenchmarkSimulationWithSolidSubdomain.y_pci))
                
        self.old_state.interpolate(("0.",  self.top_wall, "0.", temperature_string))
    
    
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.state.solution)
    
        self.adaptive_goal_form = u[0]*u[0]*self.integration_metric
        
        
    def refine_near_y_pci(self, pci_refinement_cycles = 4):   
        
        class PhaseInterface(fenics.SubDomain):
            
            def inside(self, x, on_boundary):
            
                return fenics.near(x[1], LDCBenchmarkSimulationWithSolidSubdomain.y_pci)
        
        
        phase_interface = PhaseInterface()
        
        for i in range(pci_refinement_cycles):
            
            edge_markers = fenics.EdgeFunction("bool", self.mesh)
            
            phase_interface.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
            
            self.mesh = self.mesh.child()
        
        

class HeatDrivenCavityBenchmarkSimulation(CavityBenchmarkSimulation):

    def __init__(self, mesh_size = 2):
    
        CavityBenchmarkSimulation.__init__(self, mesh_size = mesh_size)
        
        self.T_hot = 0.5
    
        self.T_cold = -self.T_hot
        
        self.boundary_conditions = [
            {"subspace": 1, "location": self.walls, "value": (0., 0.)},
            {"subspace": 2, "location": self.left_wall, "value": self.T_hot},
            {"subspace": 2, "location": self.right_wall, "value": self.T_cold}]
            
        self.rayleigh_number = 1.e6
        
        self.prandtl_number = 0.71
        
        self.timestep_size = 1.e-3
            
        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = True
        
        self.output_dir += "heat_driven_cavity/"
        
        self.adaptive_goal_tolerance = 1.e-2
        
        
    def update_initial_values(self):
    
        self.old_state.interpolate(("0.", "0.", "0.",
            "T_hot + x[0]*(T_cold - T_hot)".replace(
                "T_hot", str(self.T_hot)).replace(
                    "T_cold", str(self.T_cold))))
        
        
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.state.solution)
        
        self.adaptive_goal_form = u[0]*T*self.integration_metric
        
        
    def verify(self):
        """ Verify against the result published in \cite{wang2010comprehensive}. """
        self.verify_scalar_solution_component(
            component = 1,
            points = [((self.xmin + self.xmax)/2., y) for y in [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.]],
            verified_values = [val*self.rayleigh_number**0.5/self.prandtl_number
                for val in [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]],
            relative_tolerance = 2.e-2,
            absolute_tolerance = 1.e-2*0.0649*self.rayleigh_number**0.5/self.prandtl_number)
    

class StefanProblemBenchmarkSimulation(BenchmarkSimulation):

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
            end_time = 0.1,
            initial_uniform_cell_count = 4, 
            initial_hot_wall_refinement_cycles = 8,
            quadrature_degree = None):
    
        BenchmarkSimulation.__init__(self)
        
        initial_pci_position = \
            1./float(initial_uniform_cell_count)/2.**(initial_hot_wall_refinement_cycles - 1)
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(T_hot))
        
        initial_temperature = initial_temperature.replace("T_cold", str(T_cold))
        
        mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)

        mesh = StefanProblem.refine_near_left_boundary(mesh, initial_hot_wall_refinement_cycles)
        
        self.boundary_conditions = [
            {"subspace": 2, "location": "near(x[0],  0.)", "value": T_hot},
            {"subspace": 2, "location": "near(x[0],  1.)", "value": T_cold}]

        self.stefan_number = stefan_number = 0.045
        
        self.regularization_smoothing_parameter = regularization_smoothing_parameter
        
        self.timestep_size = timestep_size
        
        self.quadrature_degree = quadrature_degree
        
        self.model.old_state.interpolate(("0.", "0.", initial_temperature))
        
        self.end_time = end_time
        
        self.output_dir += "stefan_problem/"
    
        self.stop_when_steady = False
        
        self.adaptive_goal_tolerance = 1.e-6
        
        
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.model.state.solution)
        
        phi = self.semi_phasefield_mapping
        
        self.adaptive_goal_form = phi(T)*self.integration_metric
        
        
    def update_initial_guess(self):
        """ This test fails with the usual initial guess of w^0 = w_n, but passes with w^0 = 0. """
        self.initial_guess = ("0.", "0.", "0.")
    

    def verify(self):
        """ Verify against analytical solution. """
        self.verify_scalar_solution_component(
            component = 2,
            points = [0.00, 0.025, 0.050, 0.075, 0.10, 0.5, 1.],
            verified_values = [1.0, 0.73, 0.47, 0.20, 0.00, -0.01, -0.01],
            relative_tolerance = 2.e-2,
            absolute_tolerance = 1.e-2)
        
        
        
class ConvectionCoupledMeltingOctadecanePCMBenchmarkSimulation(CavityBenchmarkSimulation):

    def __init__(self, 
            T_hot = 1.,
            T_cold = -0.01,
            stefan_number = 0.045,
            rayleigh_number = 3.27e5,
            prandtl_number = 56.2,
            solid_viscosity = 1.e8,
            liquid_viscosity = 1.,
            timestep_size = 1.,
            initial_mesh_size = (1, 1),
            initial_hot_wall_refinement_cycles = 6,
            initial_pci_position = None,
            regularization_central_temperature = 0.01,
            regularization_smoothing_parameter = 0.025,
            end_time = 80.,
            adaptive_goal_tolerance = 1.e-5,
            quadrature_degree = 8,
            depth_3d = None):
    
        
        # Make the mesh with initial refinements near the hot wall.
        if depth_3d is None:
            
            self.spatial_dimensionality = 2
            
            Cavity.__init__(self, mesh_size = initial_mesh_size)
            
            class HotWall(fenics.SubDomain):
        
                def inside(self, x, on_boundary):
                
                    return on_boundary and fenics.near(x[0], 0.)

        
            hot_wall = HotWall()
            
            for i in range(initial_hot_wall_refinement_cycles):
                
                edge_markers = fenics.EdgeFunction("bool", self.mesh)
                
                hot_wall.mark(edge_markers, True)

                fenics.adapt(self.mesh, edge_markers)
            
                self.mesh = self.mesh.child()
            
        else:

            self.spatial_dimensionality = 3
            
            Cavity.__init__(self, mesh_size = initial_mesh_size, zmin = -depth_3d/2., zmax = depth_3d/2.)
            
            for i in range(initial_hot_wall_refinement_cycles):
            
                cell_markers = fenics.CellFunction("bool", self.mesh, False)
                
                for cell in fenics.cells(self.mesh):
                    
                    found_left_boundary = False
                    
                    for vertex in fenics.vertices(cell):
                        
                        if fenics.near(vertex.x(0), 0.):
                            
                            found_left_boundary = True
                            
                            break
                            
                    if found_left_boundary:
                        
                        cell_markers[cell] = True
                
                self.mesh = fenics.refine(self.mesh, cell_markers)
        
            
        # Set up the model.
        self.boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0.,)*self.spatial_dimensionality},
                {"subspace": 2, "location": self.left_wall, "value": T_hot},
                {"subspace": 2, "location": self.right_wall, "value": T_cold}]

        self.stefan_number = stefan_number        
        
        self.prandtl_number = prandtl_number
        
        self.rayleigh_numer = rayleigh_number
        
        self.prandtl_number = prandtl_number
        
        self.gravity = [0., -1.] + [0.,]*(self.spatial_dimensionality == 3)
                
        self.regularization_central_temperature = regularization_central_temperature
        
        self.regularization_smoothing_parameter = regularization_smoothing_parameter
        
        self.timestep_size = timestep_size
        
        self.quadrature_degree = quadrature_degree
        
        
        # Set initial values.
        if initial_pci_position == None:
        
            initial_pci_position = \
                1./float(initial_mesh_size[0])/2.**(initial_hot_wall_refinement_cycles - 1)
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(T_hot))
        
        initial_temperature = initial_temperature.replace("T_cold", str(T_cold))
        
        self.model.old_state.interpolate(
            ["0.",] + ["0.",]*self.spatial_dimensionality + [initial_temperature,])
            
        
        #
        self.adaptive_goal_tolerance = adaptive_goal_tolerance
        
        self.end_time = end_time
        
        self.stop_when_steady = False
        
        self.output_dir_suffix += "adaptive_convection_coupled_melting_octadecane_pcm/"
        
        
    def update_adaptive_goal_form(self):
    
        phi = self.semi_phasefield_mapping
        
        p, u, T = fenics.split(self.model.state.solution)
        
        self.adaptive_goal_form = phi(T)*self.integration_metric
        