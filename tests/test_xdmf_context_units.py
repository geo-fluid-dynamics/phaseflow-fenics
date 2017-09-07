from .context import phaseflow

import fenics


def test_xdmf():

    solution_file = fenics.XDMFFile('test.xdmf')


'''This test seems to fail with fenics-2016.2.0. 
I vaguely recall seeing an issue on their Bitbucket which mentions having
not always used the proper context manager style with some of their file classes.'''
def test_xdmf_context():

    with fenics.XDMFFile('test.xdmf') as solution_file:

        return


if __name__=='__main__':

    test_xdmf()

    test_xdmf_context()

