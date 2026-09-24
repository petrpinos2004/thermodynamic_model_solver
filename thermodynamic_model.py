###################################################
# Import libraries
###################################################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

###################################################
# Define helper methods
###################################################

def find_longest_true_sequence(arr):
    arr = np.array(arr, dtype=bool)
    
    max_len = 0
    best_start, best_end = -1, -1
    
    current_start = -1
    false_count = 0
    last_false_idx = -1
    
    for i, val in enumerate(arr):
        if val:
            if current_start == -1:
                current_start = i
        else:
            if current_start == -1:
                continue
            
            false_count += 1
            
            if false_count == 1:
                last_false_idx = i
            elif false_count > 1:
                current_len = i - current_start
                if current_len > max_len:
                    max_len = current_len
                    best_start, best_end = current_start, i - 1
                current_start = last_false_idx + 1
                false_count = 1
                last_false_idx = i

    if current_start != -1:
        current_len = len(arr) - current_start
        if current_len > max_len:
            max_len = current_len
            best_start, best_end = current_start, len(arr) - 1

    if best_start != -1 and best_end != -1:
        true_count = np.sum(arr[best_start : best_end + 1])
        if true_count >= 2:
            return best_start, best_end
    return None, None

###################################################
# Class thermodynamic_model
###################################################

class thermodynamic_model():

    def __init__(self, physics, maths, name, plot=False):

        self.name = name
        self.plot = plot
        self.physics = physics
        self.maths = maths

        maths['lambda'], maths['psi'], maths['d_psi_dx'] = self.base_functions(maths['N'],maths['Nx'])
        maths['PE_matrix'] = self.PE_matrix(maths['N'],maths['Nx'])

        self.solutions = {
            'solutions' : [],
            'Ra_history' : [],
            'Nu_history' : [],
            'results_list' : [],
            'header_names' : ["Time(s)"],
            't_eval' : np.linspace(0, self.physics['tmax'], 2 * self.physics['tmax'])
        }

    # --- Functions ---
    def T_top(self, t, A_temp): 
        return self.physics['T_center'] + A_temp * np.sin(2 * np.pi * self.physics['freq'] * t)

    def dT_top_dt(self, t, A_temp): 
        return A_temp * (2 * np.pi * self.physics['freq']) * np.cos(2 * np.pi * self.physics['freq'] * t)

    def base_functions(self, N, Nx):
        # --- Modes and Pre-calculated Operators ---
        lambd = (np.arange(N) + 0.5) * np.pi / self.physics['L']
            
        # Basis matrices: shape (N, Nx)
        psi = np.sin(np.outer(lambd, self.maths['x']))                 
        d_psi_dx = lambd[:, None] * np.cos(np.outer(lambd, self.maths['x']))
            
        return lambd, psi, d_psi_dx

    def PE_matrix(self, N, Nx):
        # Overlap integral and Piston Mass Matrix
        Sn = 1.0 / self.maths['lambda']
        M = np.eye(N) - (2.0 * (1.0 - 1.0 / self.physics['gamma']) / self.physics['L']**2) * np.outer(Sn, Sn)
        Minv = np.linalg.inv(M)
        return Minv

    def convection_on(self, deltaT, deltaL):
        if deltaT > 0:
            if deltaL > 0:
                Ra = (self.physics['g']*self.physics['alpha_p']*deltaT)/(self.physics['nu']*self.physics['kappa']) * deltaL**3
                Nu = self.physics['xi'] * Ra**self.physics['ratio']
                if Nu > 1:
                    return {'Nu' : Nu,
                            'Ra' : Ra}
        return {'Nu' : 1,
                'Ra' : 0}

    def system(self, t, c, A_temp):

        T = c @ self.maths['psi'] + self.T_top(t, A_temp)
        grad_T = c @ self.maths['d_psi_dx'] 
        
        # 2. Spatially varying Nusselt profile Nu(x)
        Nu_arr = np.ones(self.maths['Nx'])
        unstable_mask = (grad_T > 1e-3) & self.maths['BL_mask']
        start, end = find_longest_true_sequence(unstable_mask)
        
        if end is not None:
            if start is not None:
                deltaT = T[end] - T[start]
                deltaL = self.maths['x'][end] - self.maths['x'][start]
                convection = self.convection_on(deltaT, deltaL)
                Nu_arr[start:end] = convection['Nu']
        
        kappa_xt = self.physics['kappa'] * (Nu_arr) 
        heat_flux = kappa_xt * grad_T 
        diffusion_projection = -(2.0 / self.physics['L']) * (self.maths['d_psi_dx'] @ heat_flux) * self.maths['dx']
        PE_RHS_term = -(1.0/self.physics['gamma']) * (2.0 / (self.physics['L'] * self.maths['lambda'])) * self.dT_top_dt(t, A_temp)

        return self.maths['PE_matrix'] @ (diffusion_projection + PE_RHS_term)

    def compute_Ra_history(self, sol, A_temp):
        Ra_array = np.zeros(len(sol.t))
        Nu_array = np.ones(len(sol.t))
        # Reconstruct physical spatial profiles for all time steps
        T_all = (sol.y.T @ self.maths['psi']) + self.T_top(sol.t, A_temp)[:, None]
        grad_T_all = sol.y.T @ self.maths['d_psi_dx']
        
        for tidx in range(len(sol.t)):
            unstable_mask = (grad_T_all[tidx] > 1e-3) & self.maths['BL_mask']
            start, end = find_longest_true_sequence(unstable_mask)
            
            if start is not None and end is not None:
                deltaT = T_all[tidx, end] - T_all[tidx, start]
                deltaL = self.maths['x'][end] - self.maths['x'][start]
                convection = self.convection_on(deltaT, deltaL)
                Ra_array[tidx] = convection['Ra']
                Nu_array[tidx] = convection['Nu']
                
        return {'Ra' : Ra_array,
                'Nu' : Nu_array}

    def run(self):

        colors = ['blue', 'red', 'limegreen']
        output_filename = Path(
            rf'C:\Users\sulta\Documents\UPT  - kryogenika\starsi_veci_bakalar\thermodynamic_model\data_output\raw_simulation_data\{self.name}.txt'
        )
        output_image = Path(
            rf'C:\Users\sulta\Documents\UPT  - kryogenika\starsi_veci_bakalar\thermodynamic_model\image_output\spectral_method_convection_{self.name}.png'
        )

        plt.figure(figsize=(10, 6))
        for i, A_temp in enumerate(self.physics['amplitudes']):
            print(f"Running simulation: Amplitude = {A_temp*1000:.1f} mK")
            
            c0 = (self.physics['T_init'] - self.T_top(0, A_temp)) * (2.0 / (self.physics['L'] * self.maths['lambda']))
            
            sol = solve_ivp(
                fun=lambda t, c: self.system(t, c, A_temp),
                t_span=[0, self.physics['tmax']],
                y0=c0,
                t_eval=self.solutions['t_eval'],
                method='Radau'  # Changed to implicit solver for stiff step-function jump
            )
            self.solutions['solutions'].append(sol)
            
            T_mid = (sol.y.T @ self.maths['psi'][:, self.maths['mid_idx']]) + self.T_top(sol.t, A_temp)   
            T_b = (sol.y.T @ self.maths['psi'][:, -1]) + self.T_top(sol.t, A_temp)
            T_ref = self.T_top(sol.t, A_temp)
            
            history = self.compute_Ra_history(sol, A_temp)
            self.solutions['Ra_history'].append(history['Ra'])
            self.solutions['Nu_history'].append(history['Nu'])
            
            self.solutions['results_list'].append(T_mid)
            self.solutions['results_list'].append(T_ref)
            self.solutions['results_list'].append(T_b)
            label = f"{int(A_temp*1000)}mK"
            self.solutions['header_names'].append(f"Age_AT_{label}")
            self.solutions['header_names'].append(f"Ref_AT_{label}")
            self.solutions['header_names'].append(f"Tb_AT_{label}")

            plt.plot(sol.t, T_mid, color=colors[i], 
                    label=f'$AT={int(A_temp*1000)}$ mK')

        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [K]')
        plt.title('Thermal Response at Cell Center x = 150 mm')
        plt.legend()
        plt.savefig(output_image, dpi=300, facecolor='white')

        if self.plot:
            plt.show()
        else:
            plt.close()

        time_vector = sol.t 
        all_data = np.column_stack([time_vector] + self.solutions['results_list'])

        with open(output_filename, 'w') as f:
            f.write("spectral_simulation_check\n")
            f.write("\t".join(self.solutions['header_names']) + "\n")
            for row in all_data:
                line = "\t".join(map(str, row))
                f.write(line + "\n")
        print(f"Data successfully saved to {output_filename}")

        if self.plot:
            plt.plot(sol.t, self.solutions['Ra_history'][0])
            plt.plot(sol.t, self.solutions['Ra_history'][1])
            plt.plot(sol.t, self.solutions['Ra_history'][2])
            plt.yscale('log')
            plt.show()

            plt.plot(sol.t, self.solutions['Nu_history'][0])
            plt.plot(sol.t, self.solutions['Nu_history'][1])
            plt.plot(sol.t, self.solutions['Nu_history'][2])
            plt.show()