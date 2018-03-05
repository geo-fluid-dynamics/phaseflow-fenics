"""**benchmarks.py** applies Phaseflow to a variety of benchmark problems."""
import phaseflow
import fenics

 
class BenchmarkSimulation(phaseflow.octadecane.Simulation):
 
    def __init__(self):
    
        phaseflow.octadecane.Simulation.__init__(self)
        
        self.output_dir += "benchmark/"

    
    def run(self, verify = True):
        
        phaseflow.octadecane.Simulation.run(self)
        
        if verify:
        
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

    def __init__(self):
    
        BenchmarkSimulation.__init__(self)
        
        self.mesh_size = (2, 2)

        self.xmin = 0.
        
        self.ymin = 0.
        
        self.zmin = None
        
        self.xmax = 1.
        
        self.ymax = 1.
        
        self.zmax = None
        
        
    def validate_attributes(self):
    
        if type(self.mesh_size) is type(20):
        
            self.mesh_size = (self.mesh_size, self.mesh_size)
        
        
    def update_derived_attributes(self):
    
        BenchmarkSimulation.update_derived_attributes(self)
        
        self.left_wall = "near(x[0],  xmin)".replace("xmin", str(self.xmin))
        
        self.right_wall = "near(x[0],  xmax)".replace("xmax", str(self.xmax))
        
        self.bottom_wall = "near(x[1],  ymin)".replace("ymin",  str(self.ymin))
        
        self.top_wall = "near(x[1],  ymax)".replace("ymax", str(self.ymax))
        
        self.walls = \
                self.top_wall + " | " + self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        if len(self.mesh_size) == 3:
        
            self.back_wall = "near(x[2],  zmin)".replace("zmin",  str(self.zmin))
        
            self.front_wall = "near(x[2],  zmax)".replace("zmax", str(self.zmax))
        
            self.walls += " | " + self.back_wall + " | " + self.front_wall
        
        
    def update_mesh(self):
    
        if len(self.mesh_size) == 2:
        
            self.mesh = fenics.RectangleMesh(fenics.mpi_comm_world(), 
                fenics.Point(self.xmin, self.ymin), fenics.Point(self.xmax, self.ymax),
                self.mesh_size[0], self.mesh_size[1], "crossed")

        elif len(self.mesh_size) == 3:
        
            self.mesh = fenics.BoxMesh(fenics.mpi_comm_world(), 
                fenics.Point(self.xmin, self.ymin, self.zmin),
                fenics.Point(self.xmax, self.ymax, self.zmax),
                self.mesh_size[0], self.mesh_size[1], self.mesh_size[2])
  
  
class LidDrivenCavityBenchmarkSimulation(CavityBenchmarkSimulation):

    def __init__(self):
    
        CavityBenchmarkSimulation.__init__(self)
        
        self.relative_tolerance = 3.e-2
        
        self.absolute_tolerance = 1.e-2
        
        self.timestep_size = 1.e12
        
        self.liquid_viscosity = 0.01
        
        self.end_time = 0. + self.timestep_size
        
        self.output_dir += "lid_driven_cavity/"
        
        self.adaptive_goal_tolerance = 1.e-4
  
    
    def update_derived_attributes(self):
  
        CavityBenchmarkSimulation.update_derived_attributes(self)
        
        self.fixed_walls = self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        self.boundary_conditions = [
            {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
            {"subspace": 1, 
             "location": self.fixed_walls, 
             "value": (0., 0.)}]
  
  
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
    def __init__(self):
    
        LidDrivenCavityBenchmarkSimulation.__init__(self)
        
        self.ymin = -0.25
        
        self.y_pci = 0.
        
        self.mesh_size = (4, 5)
        
        self.pci_refinement_cycles = 4
        
        self.timestep_size = 20.
        
        self.relative_tolerance = 8.e-2  # @todo This is quite large.
        
        self.absolute_tolerance = 2.e-2
        
        self.adaptive_goal_tolerance = 1.e-5
                
        self.prandtl_number = 1.e16
        
        self.liquid_viscosity = 0.01
        
        self.solid_viscosity = 1.e6
        
        self.stefan_number = 1.e16
        
        self.regularization_central_temperature = -0.01
        
        self.regularization_smoothing_parameter = 0.01
        
        self.end_time = 2.*self.timestep_size
        
        self.quadrature_degree = 3
        
        self.output_dir += "with_solid_subdomain/"
        
    
    def update_derived_attributes(self):
    
        LidDrivenCavityBenchmarkSimulation.update_derived_attributes(self)
    
        self.boundary_conditions = [
                {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
                {"subspace": 1, "location": self.fixed_walls, "value": (0., 0.)}]
    
    
    def update_mesh(self):
        
        LidDrivenCavityBenchmarkSimulation.update_mesh(self)

        y_pci = self.y_pci
        
        class PhaseInterface(fenics.SubDomain):
            
            def inside(self, x, on_boundary):
            
                return fenics.near(x[1], y_pci)
        
        
        phase_interface = PhaseInterface()
        
        for i in range(self.pci_refinement_cycles):
            
            edge_markers = fenics.EdgeFunction("bool", self.mesh)
            
            phase_interface.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
            
            self.mesh = self.mesh.child()
    
    
    def update_initial_values(self):
    
        temperature_string = "1. - 2.*(x[1] <= y_pci)".replace("y_pci", str(self.y_pci))
                
        self.old_state.interpolate(("0.",  self.top_wall, "0.", temperature_string))
    
    
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.state.solution)
    
        self.adaptive_goal_form = u[0]*u[0]*self.integration_metric
            

class HeatDrivenCavityBenchmarkSimulation(CavityBenchmarkSimulation):

    def __init__(self):
    
        CavityBenchmarkSimulation.__init__(self)
        
        self.T_hot = 0.5
    
        self.T_cold = -self.T_hot
        
        self.solid_viscosity = 1.
        
        self.stefan_number = 1.e32
        
        self.rayleigh_number = 1.e6
        
        self.prandtl_number = 0.71
        
        self.timestep_size = 1.e-3
        
        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_residual = True
        
        self.output_dir += "heat_driven_cavity/"
        
        self.adaptive_goal_tolerance = 4.e-2
        
        
    def update_derived_attributes(self):
    
        CavityBenchmarkSimulation.update_derived_attributes(self)
        
        self.boundary_conditions = [
            {"subspace": 1, "location": self.walls, "value": (0., 0.)},
            {"subspace": 2, "location": self.left_wall, "value": self.T_hot},
            {"subspace": 2, "location": self.right_wall, "value": self.T_cold}]
            
        self.initial_temperature = "T_hot + x[0]*(T_cold - T_hot)"
        
        self.initial_temperature = self.initial_temperature.replace("T_hot", str(self.T_hot))
        
        self.initial_temperature = self.initial_temperature.replace("T_cold", str(self.T_cold))
        
        
    def update_initial_values(self):
    
        self.old_state.interpolate(("0.", "0.", "0.", self.initial_temperature))
        
        
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
            relative_tolerance = 1.e-2,
            absolute_tolerance = 1.e-2*0.0649*self.rayleigh_number**0.5/self.prandtl_number)
    

class StefanProblemBenchmarkSimulation(BenchmarkSimulation):

    def __init__(self):
    
        BenchmarkSimulation.__init__(self)
        
        self.initial_uniform_cell_count = 4
        
        self.initial_hot_boundary_refinement_cycles = 8
        
        self.initial_pci_position = None
        
        self.T_hot = 1.
        
        self.T_cold = -0.01
        
        self.stefan_number = 0.045
        
        self.gravity = (0.,)
        
        self.regularization_smoothing_parameter = 0.005
        
        self.timestep_size = 1.e-3
        
        self.quadrature_degree = None
        
        self.end_time = 0.1
        
        self.output_dir += "stefan_problem/"
    
        self.stop_when_steady = False
        
        self.adaptive_goal_tolerance = 1.e-6
        
        
    def update_derived_attributes(self):
    
        BenchmarkSimulation.update_derived_attributes(self)
        
        self.boundary_conditions = [
            {"subspace": 2, "location": "near(x[0],  0.)", "value": self.T_hot},
            {"subspace": 2, "location": "near(x[0],  1.)", "value": self.T_cold}]
        
        if self.initial_pci_position is None:
        
            initial_pci_position = 1./float(self.initial_uniform_cell_count)/2.**( \
                self.initial_hot_boundary_refinement_cycles - 1)
                
        else:
        
            initial_pci_position = self.initial_pci_position
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(self.T_hot))
        
        self.initial_temperature = initial_temperature.replace("T_cold", str(self.T_cold))
        
    
    def update_mesh(self):
    
        self.mesh = fenics.UnitIntervalMesh(self.initial_uniform_cell_count)
        
        for i in range(self.initial_hot_boundary_refinement_cycles):
            
            cell_markers = fenics.CellFunction("bool", self.mesh)
            
            cell_markers.set_all(False)
            
            for cell in fenics.cells(self.mesh):
                
                found_left_boundary = False
                
                for vertex in fenics.vertices(cell):
                    
                    if fenics.near(vertex.x(0), 0.):
                        
                        found_left_boundary = True
                        
                        break
                        
                if found_left_boundary:
                    
                    cell_markers[cell] = True
                    
                    break # There should only be one such point in 1D.
                    
            self.mesh = fenics.refine(self.mesh, cell_markers)
    
    
    def update_initial_values(self):
        
        self.old_state.interpolate(("0.", "0.", self.initial_temperature))
        
        
    def update_adaptive_goal_form(self):
    
        p, u, T = fenics.split(self.state.solution)
        
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

    def __init__(self):
    
        CavityBenchmarkSimulation.__init__(self)
        
        self.T_hot = 1.
        
        self.T_cold = -0.01
        
        self.stefan_number = 0.045      
        
        self.prandtl_number = 56.2
        
        self.rayleigh_number = 3.27e5
        
        self.gravity = (0., -1.)
        
        self.solid_viscosity = 1.e8
        
        self.liquid_viscosity = 1.
        
        self.regularization_central_temperature = 0.01
        
        self.regularization_smoothing_parameter = 0.025
        
        self.timestep_size = 1.
        
        self.quadrature_degree = 8
        
        self.initial_hot_wall_refinement_cycles = 6
        
        self.initial_mesh_size = (1, 1)
        
        self.initial_pci_position = None
        
        self.adaptive_goal_tolerance = 1.e-5
        
        self.end_time = 80.
        
        self.stop_when_steady = False
        
        self.output_dir += "adaptive_convection_coupled_melting_octadecane_pcm/"
        
        
    def update_derived_attributes(self):
    
        CavityBenchmarkSimulation.update_derived_attributes(self)
        
        self.boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": self.T_hot},
                {"subspace": 2, "location": self.right_wall, "value": self.T_cold}]
                
        if self.initial_pci_position == None:
        
            initial_pci_position = \
                1./float(self.initial_mesh_size[0])/2.**(self.initial_hot_wall_refinement_cycles - 1)
        
        else:
        
            initial_pci_position = 0. + self.initial_pci_position
        
        initial_temperature = "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace("T_hot", str(self.T_hot))
        
        initial_temperature = initial_temperature.replace("T_cold", str(self.T_cold))
        
        self.initial_temperature = initial_temperature
        
    
    def update_mesh(self):
    
        CavityBenchmarkSimulation.update_mesh(self)
        
        class HotWall(fenics.SubDomain):
    
            def inside(self, x, on_boundary):
            
                return on_boundary and fenics.near(x[0], 0.)

    
        hot_wall = HotWall()
        
        for i in range(self.initial_hot_wall_refinement_cycles):
            
            edge_markers = fenics.EdgeFunction("bool", self.mesh)
            
            hot_wall.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
        
            self.mesh = self.mesh.child()
            
    
    def update_initial_values(self):
        
        self.old_state.interpolate(("0.", "0.", "0.", self.initial_temperature))
        
        
    def update_adaptive_goal_form(self):
    
        phi = self.semi_phasefield_mapping
        
        p, u, T = fenics.split(self.state.solution)
        
        self.adaptive_goal_form = phi(T)*self.integration_metric
        
     
class CCMOctadecanePCMBenchmarkSimulation3D(CavityBenchmarkSimulation):

    def __init__(self):
    
        ConvectionCoupledMeltingOctadecanePCMBenchmarkSimulation.__init__(self)
        
        self.mesh_size = (1, 1, 1)
        
        self.gravity = (0., -1., 0.)
        
        self.boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": self.T_hot},
                {"subspace": 2, "location": self.right_wall, "value": self.T_cold}]
        
        self.depth_3d = 0.5
        
        
    def update_mesh(self):

        self.zmin = -depth_3d/2.
        
        self.zmax = depth_3d/2.
        
        for i in range(self.initial_hot_wall_refinement_cycles):
        
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
            
            
    def update_initial_values(self):
        
        self.old_state.interpolate((0., 0., 0., 0., self.initial_temperature))
            