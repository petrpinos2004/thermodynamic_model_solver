import python_core.thermodynamic_model as mother
import python_core.data_processing as father
import numpy as np

####################################################
# Define Simulation class
####################################################

class Simulation(mother.thermodynamic_model):

    def __init__(self, name='h431', plot=False):

        physics = {
            'L' : 0.3,  
            'kappa' : 2.02e-7,      # Thermal diffusivity
            'gamma' : 1.94,         # Gamma
            'nu' : 1.67e-7,         # Dynamic viscosity
            'g' : 9.81,             # Gravity
            'alpha_p' : 2.23,       # Expansivity
            'ratio' : 1/3,          # Nu-Ra exponent
            'xi' : 0.06,             # Nu-Ra prefactor
            'delta_diff' : [0.00441, 0.00253, 0.00146], # Stokes diffusion layer thickness
            
            'T_init' : 5.0,         # Starting fluid temp (K)
            'T_center' : 5.1,       # Modulation temp center (K)
            'freq' : [0.0033, 0.01, 0.03],          # Modulation frequency
            'amplitudes' : [0.050],  # Modulation for simulation sweep
            'tmax': [1800, 1000, 300],          # Total simulation time
        }

        maths = {
            'N': 100,              # Sine modes
            'Nx': 500,             # Spatial grid points
        }

        maths['x'] = np.linspace(0, physics['L'], maths['Nx'])
        maths['dx'] = maths['x'][1] - maths['x'][0]
        maths['mid_idx'] = maths['Nx'] // 2

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

h431 = Simulation(plot=True)
h431.run()
Data(h431.name, h431.physics['freq'], h431.plot)

