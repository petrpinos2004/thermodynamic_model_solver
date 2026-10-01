import python_core.thermodynamic_model as mother
import python_core.data_processing as father
import numpy as np

####################################################
# Define Simulation class
####################################################

class Simulation(mother.thermodynamic_model):

    def __init__(self, name='h430', plot=False):

        physics = {
            'L' : 0.3,  
            'kappa' : 2.37e-7,      # Thermal diffusivity
            'gamma' : 1.9,         # Gamma
            'nu' : 1.91e-7,         # Dynamic viscosity
            'g' : 9.81,             # Gravity
            'alpha_p' : 0.26,       # Expansivity
            'ratio' : 1/3,          # Nu-Ra exponent
            'xi' : 0.06,             # Nu-Ra prefactor
            'delta_diff' : [0.00409], # Stokes diffusion layer thickness
            
            'T_init' : 5.0,         # Starting fluid temp (K)
            'T_center' : 5.1,       # Modulation temp center (K)
            'freq' : [0.0045],          # Modulation frequency
            'amplitudes' : [0.025, 0.050, 0.075],  # Modulation for simulation sweep
            'tmax': [1800],          # Total simulation time
        }

        maths = {
            'N': 200,              # Sine modes
            'Nx': 5000,             # Spatial grid points
        }

        super().__init__(physics, maths, name, plot)

####################################################
# Define Data class
####################################################

class Data(father.Data):

    def __init__(self, name, f0, plot=False):

        super().__init__(name, f0, plot)

####################################################
# Call
####################################################

h430 = Simulation()
h430.run()
Data(h430.name, h430.physics['freq'], h430.plot)

