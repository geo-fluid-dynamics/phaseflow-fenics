""" **test_abstract_phasechange_simulation.py** tests **phaseflow/abstract_phasechange_simulation.py**,
Phaseflow's module for the natural and compositional convection of binary alloys."""
import phaseflow
import fenics

    
class LidDrivenCavityBenchmarkSimulation(phaseflow.abstract_phasechange_simulation.AbstractPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            setup_solver = True):
        
        class Lid(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[1], 1.)
                
        self.lid = Lid()
        
        class FixedWalls(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and (not fenics.near(x[1], 1.))
                
        self.fixed_walls = FixedWalls()
        
        class BottomWall(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[1], 0.)
                
        self.bottom_wall = BottomWall()
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
        self.timestep_size = 1.e12
        
        self.temperature_rayleigh_number.assign(0.)
        
        self.concentration_rayleigh_number.assign(0.)
        
        self.schmidt_number.assign(1.e32)
        
        self.prandtl_number.assign(1.)
        
        """ Disable effects of phase-change. 
        Oddly enough, simply specifying an arbitrarily low melting temperature
        results in the Newton solver returning NaN's, e.g. the following did not work:
        
            self.regularization_central_temperature.assign(1.e-32)
            
        As a reminder of this for future development, let's assign something that works here.
        """
        self.pure_liquidus_temperature.assign(0.)
        
        self.liquidus_slope.assign(0.)
        
        self.regularization_central_temperature_offset.assign(0.)
        
        self.liquid_viscosity.assign(0.01)
        
        self.solid_viscosity.assign(self.liquid_viscosity)
        
        self.stefan_number.assign(1.e32)
        
    def coarse_mesh(self):
        
        return fenics.UnitSquareMesh(4, 4)
        
    def initial_mesh(self):
        
        return self.coarse_mesh()
        
    def initial_values(self):
        
        p0, u0_0, u0_1, T0, C0 = "0.", "1.", "0.", "0.", "0."
        
        w0 = fenics.interpolate(
            fenics.Expression(
                (p0, u0_0, u0_1, T0, C0),
                element = self.element()),
            self.function_space)
        
        return w0
            
    def boundary_conditions(self):
    
        return [
            fenics.DirichletBC(
                self.function_space.sub(1), 
                (1., 0.), 
                self.lid),
            fenics.DirichletBC(
                self.function_space.sub(1), 
                (0., 0.), 
                self.fixed_walls)]
            
    def bottom_wall_shear_integrand(self):
    
        nhat = fenics.FacetNormal(self.mesh)
    
        p, u, T, C = fenics.split(self.solution)
        
        bottom_wall_id = 2
        
        mesh_function = fenics.MeshFunction(
            "size_t", 
            self.mesh, 
            self.mesh.topology().dim() - 1)
        
        self.bottom_wall.mark(mesh_function, bottom_wall_id)
        
        dot, grad = fenics.dot, fenics.grad
        
        ds = fenics.ds(
            domain = self.mesh, 
            subdomain_data = mesh_function, 
            subdomain_id = bottom_wall_id)
        
        return dot(grad(u[0]), nhat)*ds
        
    def adaptive_goal(self):
        """ The default refinement algorithm has a 'known bug' which prevents us from 
        integrating a goal over a subdomain, referenced in the following:
        - https://fenicsproject.org/qa/6719/using-adapt-on-a-meshfunction-looking-for-a-working-example/
        - https://bitbucket.org/fenics-project/dolfin/issues/105/most-refinement-algorithms-do-not-set
        So we follow their advice and set the following algorithim.
        """
        fenics.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        
        return self.bottom_wall_shear_integrand()
    

def test_lid_driven_cavity__ci__():

    verified_bottom_wall_shear = 0.272
    
    tolerance = 0.001
    
    sim = LidDrivenCavityBenchmarkSimulation()
    
    sim.assign_initial_values()
    
    sim.solve(goal_tolerance = tolerance)
    
    sim.advance()
    
    sim.solve(goal_tolerance = tolerance)
    
    integrate = fenics.assemble
    
    bottom_wall_shear = integrate(sim.bottom_wall_shear_integrand())
    
    print("Integrated bottom wall shear = " + str(bottom_wall_shear))
    
    assert(abs(bottom_wall_shear - verified_bottom_wall_shear) < tolerance)

    
def test_write_solution_with_velocity_field_for_paraview_streamlines():
    """ Sometimes ParaView does not plot streamlines properly 
    for a mixed finite element solution written to a fenics.XDMFFile.
    phaseflow.simulation.SolutionFile sets a property of fenics.XDMFFile which fixes this. 
    Automatically verifying this behavior is not practical; but this test writes a file that
    can be checked manually. """
    sim = LidDrivenCavityBenchmarkSimulation()
    
    sim.assign_initial_values()
    
    sim.solve(goal_tolerance = 0.001)
    
    with phaseflow.abstract_simulation.SolutionFile("test__lid_driven_cavity.xdmf") as file:
    
        sim.write_solution(file)
