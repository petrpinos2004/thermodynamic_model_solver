import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from pathlib import Path

# --- Parameters ---
L = 0.3  
kappa = 2.25e-7      # Thermal diffusivity
gamma = 1.92         # Gamma
nu = 1.83e-7         # Dynamic viscosity
g = 9.81             # Gravity
alpha_p = 1.96       # Expansivity
ratio = 1/3          # Nu-Ra exponent
xi = 0.06             # Nu-Ra prefactor
delta_diff = 0.00268 # Stokes diffusion layer thickness

T_init = 5.0         # Starting fluid temp (K)
T_center = 5.1       # Modulation temp center (K)
freq = 0.01          # Modulation frequency
amplitudes = [0.025, 0.050, 0.075] 
colors = ['blue', 'red', 'limegreen']

N = 100              # Sine modes 
Nx = 500             # Spatial grid points
tmax = 1500          # Total simulation time

# --- Storage for File Output ---
results_list = []
header_names = ["Time(s)"]

x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
mid_idx = Nx // 2
BL_mask = x <= 5*delta_diff # Mask to strictly restrict Nu to the boundary layer


# --- Functions ---
def T_top(t, A_temp): 
    return T_center + A_temp * np.sin(2 * np.pi * freq * t)
    
def dT_top_dt(t, A_temp): 
    return A_temp * (2 * np.pi * freq) * np.cos(2 * np.pi * freq * t)

import numpy as np

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

def convection_on(deltaT, deltaL):
    if deltaT > 0:
        if deltaL > 0:
            Ra = (g*alpha_p*deltaT)/(nu*kappa) * deltaL**3
            Nu = xi * Ra**ratio
            if Nu > 1:
                return {'Nu' : Nu,
                        'Ra' : Ra}
    return {'Nu' : 1,
            'Ra' : 0}

def system(t, c, A_temp):

    T = c @ base_functions['psi'] + T_top(t, A_temp)
    grad_T = c @ base_functions['d_psi_dx'] 
    
    # 2. Spatially varying Nusselt profile Nu(x)
    Nu_arr = np.ones(Nx)
    unstable_mask = (grad_T > 1e-3) & BL_mask
    start, end = find_longest_true_sequence(unstable_mask)
    
    if end is not None:
        if start is not None:
            deltaT = T[end] - T[start]
            deltaL = x[end] - x[start]
            convection = convection_on(deltaT, deltaL)
            Nu_arr[start:end] = convection['Nu']
    
    kappa_xt = kappa * (Nu_arr) 
    heat_flux = kappa_xt * grad_T 
    diffusion_projection = -(2.0 / L) * (base_functions['d_psi_dx'] @ heat_flux) * dx
    PE_RHS_term = -(1.0/gamma) * (2.0 / (L * base_functions['lambda'])) * dT_top_dt(t, A_temp)

    return PE_matrix @ (diffusion_projection + PE_RHS_term)

def compute_Ra_history(sol, A_temp):
    Ra_array = np.zeros(len(sol.t))
    Nu_array = np.ones(len(sol.t))
    # Reconstruct physical spatial profiles for all time steps
    T_all = (sol.y.T @ base_functions['psi']) + T_top(sol.t, A_temp)[:, None]
    grad_T_all = sol.y.T @ base_functions['d_psi_dx']
    
    for tidx in range(len(sol.t)):
        unstable_mask = (grad_T_all[tidx] > 1e-3) & BL_mask
        start, end = find_longest_true_sequence(unstable_mask)
        
        if start is not None and end is not None:
            deltaT = T_all[tidx, end] - T_all[tidx, start]
            deltaL = x[end] - x[start]
            convection = convection_on(deltaT, deltaL)
            Ra_array[tidx] = convection['Ra']
            Nu_array[tidx] = convection['Nu']
            
    return {'Ra' : Ra_array,
            'Nu' : Nu_array}

def base_functions(N,Nx):
    # --- Modes and Pre-calculated Operators ---
    lambd = (np.arange(N) + 0.5) * np.pi / L
    
    # Basis matrices: shape (N, Nx)
    psi = np.sin(np.outer(lambd, x))                 
    d_psi_dx = lambd[:, None] * np.cos(np.outer(lambd, x))
    
    return {'lambda'  : lambd,
            'psi'     : psi,
            'd_psi_dx':d_psi_dx
           }

def PE_matrix(N,Nx):
    # Overlap integral and Piston Mass Matrix
    Sn = 1.0 / base_functions['lambda']
    M = np.eye(N) - (2.0 * (1.0 - 1.0 / gamma) / L**2) * np.outer(Sn, Sn)
    Minv = np.linalg.inv(M)
    return Minv

base_functions = base_functions(N,Nx)
PE_matrix = PE_matrix(N,Nx)

# --- Run Simulation Sweep ---
t_eval = np.linspace(0, tmax, 2 * tmax)
plt.figure(figsize=(10, 6))

solutions = [] # Store solution objects for post-processing/verification
Ra_history = []
Nu_history = []

for i, A_temp in enumerate(amplitudes):
    print(f"Running simulation: Amplitude = {A_temp*1000:.1f} mK")
    
    c0 = (T_init - T_top(0, A_temp)) * (2.0 / (L * base_functions['lambda']))
    
    sol = solve_ivp(
        fun=lambda t, c: system(t, c, A_temp),
        t_span=[0, tmax],
        y0=c0,
        t_eval=t_eval,
        method='Radau'  # Changed to implicit solver for stiff step-function jump
    )
    solutions.append(sol)
    
    T_mid = (sol.y.T @ base_functions['psi'][:, mid_idx]) + T_top(sol.t, A_temp)   
    T_b = (sol.y.T @ base_functions['psi'][:, -1]) + T_top(sol.t, A_temp)
    T_ref = T_top(sol.t, A_temp)
    
    history = compute_Ra_history(sol, A_temp)
    Ra_history.append(history['Ra'])
    Nu_history.append(history['Nu'])
    
    results_list.append(T_mid)
    results_list.append(T_ref)
    results_list.append(T_b)
    label = f"{int(A_temp*1000)}mK"
    header_names.append(f"Age_AT_{label}")
    header_names.append(f"Ref_AT_{label}")
    header_names.append(f"Tb_AT_{label}")

    plt.plot(sol.t, T_mid, color=colors[i], 
             label=f'$AT={int(A_temp*1000)}$ mK')

plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.title('Thermal Response at Cell Center x = 150 mm')
plt.legend()
plt.savefig('spectral_method_convection_h432.png', dpi=300, facecolor='white')
plt.show()

# --- Save Results to File ---
output_filename = Path(
    r"C:\Users\sulta\Documents\UPT  - kryogenika\starsi_veci_bakalar\automaticke_zpracovani_dat\simulation_results_h432_kappa_my2.txt"
)
time_vector = sol.t 
all_data = np.column_stack([time_vector] + results_list)

with open(output_filename, 'w') as f:
    f.write("spectral_simulation_check\n")
    f.write("\t".join(header_names) + "\n")
    for row in all_data:
        line = "\t".join(map(str, row))
        f.write(line + "\n")
print(f"Data successfully saved to {output_filename}")

plt.plot(sol.t, Ra_history[0])
plt.plot(sol.t, Ra_history[1])
plt.plot(sol.t, Ra_history[2])
plt.yscale('log')
plt.show()

plt.plot(sol.t, Nu_history[0])
plt.plot(sol.t, Nu_history[1])
plt.plot(sol.t, Nu_history[2])
plt.show()