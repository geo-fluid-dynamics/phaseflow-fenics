 class Cavity():

    def __init__(self, grid_size = 20):
    
        self.mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), grid_size, grid_size)
    
        self.left_wall = "near(x[0],  0.)"
        
        self.right_wall = "near(x[0],  1.)"
        
        self.bottom_wall = "near(x[1],  0.)"
        
        self.top_wall = "near(x[1],  1.)"
        
        
class LidDrivenCavity(Cavity):

    def __init__(self, grid_size = 20):
    
        Cavity.__init__(self, grid_size)
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh, 
            buoyancy_function = pure.ConstantFunctionOfTemperature(0.),
            semi_phasefield_mapping = pure.ConstantFunctionOfTemperature(0.),
            liquid_viscosity = 0.01,
            time_step_size = 1.e12)
        
        self.model.state.solution = fenics.interpolate(
            fenics.Expression(("0.", lid, "0.", "1."), element = model.element), 
            model.function_space)

        problem = pure.Problem(
            model,
            velocity_boundary_conditions = 
                {self.top_wall: (1., 0.),
                self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall: (0., 0.)},
            time_step_size = self.end_time)
            
    
    def verify(self):
        """ Verify against ghia1982 """
        data = {'Re': 100, 'x': 0.5, 
            'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 
            'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, 
                -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
        
        bbt = self.model.mesh().bounding_box_tree()
        
        for i, true_ux in enumerate(data['ux']):
        
            p = fenics.Point(data['x'], data['y'][i])
            
            if bbt.collides_entity(p):
            
                values = solution(p)
                
                ux = values[1]
                
                assert(abs(ux - true_ux) < 2.e-2)
                
                
    def run():

        time_stepper = phaseflow.TimeStepper(output_dir = "output/benchmarks/lid_driven_cavity")
        
        time_stepper.run_until(problem, end_time = self.model.time_step_size)
            
        self.verify()
    
    
if __name__=='__main__':

    LidDrivenCavity().run()
    