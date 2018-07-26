""" **plotting.py** defines utility functions for plotting meshes and finite element functions. """
import fenics
import matplotlib


def plot(f):
    """ This patches `fenics.plot` which is incorrect for functions on refind 1D meshes. 
    
    See https://bitbucket.org/fenics-project/dolfin/issues/1029/plotting-1d-function-incorrectly-ignores
    """
    if (type(f) == fenics.Function) and (f.function_space().mesh().topology().dim() == 1):

        mesh = f.function_space().mesh()

        C = f.compute_vertex_values(mesh)

        X = list(mesh.coordinates()[:, 0])

        sorted_C = [c for _,c in sorted(zip(X, C))]

        matplotlib.pyplot.plot(sorted(X), sorted_C)

    else:

        fenics.plot(f)
        