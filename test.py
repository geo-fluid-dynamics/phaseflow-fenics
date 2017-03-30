import danaila_natural_convection
#import newton_lid_driven_cavity
#import newton_steady_lid_driven_cavity

# wang2010 Figure 4.a shows that M=20 recovered the correct peak velocity, while M=40 recovers the inflection point at the center. danaila2014newton shows the result for M=80.

#danaila_natural_convection.run(mesh_M=20, time_step_size=1.e-4, output_dir='M20')

#danaila_natural_convection.run(mesh_M=40, time_step_size=1.e-4, output_dir='M40')

danaila_natural_convection.run(linearize=True, mesh_M=20, time_step_size=1.e-3, output_dir='linearized_M20')

#danaila_natural_convection.run(mesh_M=20, time_step_size=1.e-3, final_time=0., output_dir='t0_M20')

#newton_lid_driven_cavity.run(mesh_M=32, time_step_size=1.e-3, final_time=1.)

#newton_lid_driven_cavity.run(linearize=True, output_dir='output_linearized_nldc', mesh_M=32, time_step_size=1.e-3, final_time=1.)

#newton_steady_lid_driven_cavity.run(mesh_M=32)