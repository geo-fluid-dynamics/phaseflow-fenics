FROM quay.io/fenicsproject/stable:2017.1.0

RUN pip3 install h5py

RUN pip3 install sphinx sphinx-autobuild

RUN git clone https://github.com/geo-fluid-dynamics/phaseflow-fenics.git

RUN cd phaseflow-fenics && pip3 install . && cd ..
