import danaila_natural_convection

# wang2010 Figure 4.a shows that M=20 recovered the correct peak velocity, while M=40 recovers the inflection point at the center. danaila2014newton shows the result for M=80.

danaila_natural_convection.run(mesh_M=20, time_step_size=1.e-4, output_dir='M20')

#danaila_natural_convection.run(mesh_M=40, time_step_size=0.5e-4, output_dir='M40')

#danaila_natural_convection.run(linearize=True, mesh_M=20, time_step_size=1.e-3, output_dir='linearize_M20')

#danaila_natural_convection.run(mesh_M=20, time_step_size=1.e-3, final_time=0., output_dir='t0_M20')