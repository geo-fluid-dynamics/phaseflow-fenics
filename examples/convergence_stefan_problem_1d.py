import fenics
import phaseflow
import scipy.optimize as opt

   
def extract_pci_position(w):

    def theta(x):
    
        wval = w(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.newton(theta, 0.1)
    
    return pci_pos
    
    
def stefan_problem_solidify(Ste = 0.125,
    theta_h = 0.01,
    theta_c = -1.,
    theta_f = 0.,
    r = 0.01,
    dt = 0.01,
    end_time = 1.,
    nlp_absolute_tolerance = 1.e-4,
    initial_uniform_cell_count = 100,
    automatic_jacobian = False):

    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)

    w, mesh = phaseflow.run(
        output_dir = 'output/convergence_stefan_problem_solidify/dt'+str(dt)+
            '/dx'+str(1./float(initial_uniform_cell_count))+'/',
        Pr = 1.,
        Ste = Ste,
        g = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_c)+" - "+str(theta_h)+")*near(x[0],  0.) + "+str(theta_h)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'T_f': theta_f, 'r': r},
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        end_time = end_time,
        time_step_bounds = dt,
        output_times = ('end',),
        automatic_jacobian = automatic_jacobian)
        
    return w
  

def convergence_stefan_problem_1d():

    phaseflow.helpers.mkdir_p('output/convergence_stefan_problem_solidify/')
    
    with open('output/convergence_stefan_problem_solidify/convergence.txt',
            'a+') as file:
    
        file.write("dt,dx,pci_pos\n")
    
        nx = 800
        
        for nt in [25, 50, 100, 200]:
        
            dt = 1./float(nt)
            
            dx = 1./float(nx)
        
            w = stefan_problem_solidify(dt = 1./float(nt), initial_uniform_cell_count = nx, )
        
            pci_pos = extract_pci_position(w)
            
            file.write(str(dt)+","+str(dx)+","+str(pci_pos)+"\n")
        
        nt = 200

        for nx in [100, 200, 400]:
        
            dt = 1./float(nt)
            
            dx = 1./float(nx)
        
            w = stefan_problem_solidify(dt = 1./float(nt), initial_uniform_cell_count = nx, )
        
            pci_pos = extract_pci_position(w)
            
            file.write(str(dt)+","+str(dx)+","+str(pci_pos)+"\n")

            
if __name__=='__main__':
    
    convergence_stefan_problem_1d()
