""" **heated_cavity_phasechange_simulation.py** applies Phaseflow to benchmark problems 
on the unit square with hot and cold vertical walls, and adiabatic horizontal walls. 
"""
import phaseflow
import fenics


class AbstractHeatedCavityPhaseChangeSimulation(phaseflow.abstract_phasechange_simulation.AbstractPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            initial_uniform_gridsize = 20,
            setup_solver = True):
    
        self.hot_wall_temperature = fenics.Constant(1., name = "T_h")
        
        self.cold_wall_temperature = fenics.Constant(-0.01, name = "T_c")
        
        self.initial_concentration = fenics.Constant(1., name = "C0")
   
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
        
        self.initial_uniform_gridsize = initial_uniform_gridsize
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
        
    def coarse_mesh(self):
        
        M = self.initial_uniform_gridsize
        
        return fenics.UnitSquareMesh(M, M)
    
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
        
    def deepcopy(self):
    
        sim = super().deepcopy()
        
        sim.hot_wall_temperature.assign(self.hot_wall_temperature)
        
        sim.cold_wall_temperature.assign(self.cold_wall_temperature)
        
        sim.initial_concentration.assign(self.initial_concentration)
        
        sim.hot_wall = self._HotWall()
        
        sim.cold_wall = self._ColdWall()
        
        sim.walls = self._Walls()
        
        return sim
        
    def cold_wall_heat_flux_integrand(self):
    
        nhat = fenics.FacetNormal(self.mesh.leaf_node())
    
        p, u, T, C = fenics.split(self.solution.leaf_node())
        
        mesh_function = fenics.MeshFunction(
            "size_t", 
            self.mesh.leaf_node(), 
            self.mesh.topology().dim() - 1)
        
        cold_wall_id = 2
        
        self.cold_wall.mark(mesh_function, cold_wall_id)
        
        dot, grad = fenics.dot, fenics.grad
        
        ds = fenics.ds(
            domain = self.mesh.leaf_node(), 
            subdomain_data = mesh_function, 
            subdomain_id = cold_wall_id)
        
        return dot(grad(T), nhat)*ds
        