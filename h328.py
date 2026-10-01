import python_core.thermodynamic_model as mother
import python_core.data_processing as father
import numpy as np

####################################################
# Define Simulation class
####################################################

class Simulation(mother.thermodynamic_model):

    def __init__(self, name='h328', plot=False):

        physics = {
            'L' : 0.3,  
            'kappa' : 2.2e-8,       # Thermal diffusivity
            'gamma' : 5.58,          # Gamma
            'nu' : 4.8e-8,          # Dynamic viscosity
            'g' : 9.81,              # Gravity
            'alpha_p' : 1.11,         # Expansivity
            'ratio' : 1/3,           # Nu-Ra exponent
            'xi' : 0.06,             # Nu-Ra prefactor
            'delta_diff' : [0.00083], # Stokes diffusion layer thickness
            
            'T_init' : 5.24,         # Starting fluid temp (K)
            'T_center' : 5.31,       # Modulation temp center (K)    
            'freq' : [0.01],         # Modulation frequency
            'amplitudes' : [0.050],  # Modulation for simulation sweep
            'tmax': [1000],          # Total simulation time
        }

        maths = {
            'N': 300,                # Sine modes
            'Nx': 10000,             # Spatial grid points
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

h328 = Simulation()
h328.run()
Data(h328.name, h328.physics['freq'], h328.plot)

