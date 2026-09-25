import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import re

# These will be set during initialization
sweep = None
parametric_sweep = None
studies = None
header_line = None
rough_file = None
columns = None
modulation = None
date = None
is_cartesian = True
r = None
z = None
criterion = None

def starting_procedure(file):

    global sweep, parametric_sweep, studies, header_line, rough_file, columns, r, z, modulation, \
    date, is_cartesian, criterion

    with open(file, 'r') as f:
        first_line = f.readline().strip()
        criterion = "spectral_simulation_check" in first_line.lower()

    if criterion:
        with open(file, 'r') as f:
            f.readline()  # Skip identifier
            header_line = f.readline().strip().split('\t')  
            sweep = header_line[1].split('_')[1]
            parametric_sweep = sorted(list(set(int(''.join(filter(str.isdigit, h.split('_')[-1]))) for h in header_line if '_' in h)))
            rough_file = np.array([[float(x) for x in line.split('\t')] for line in f if line.strip()])
            studies = len(parametric_sweep)

        return sweep, parametric_sweep, studies, header_line, modulation, date, is_cartesian, criterion
        
    else:
        # Read metadata
        modulation = None
        date = None
        with open(file, encoding="utf-8") as f:
            for idx in range(9):
                if idx == 8:
                    header_line = f.readline().strip()
                else:
                    line = f.readline().strip()
                    if line.startswith('% Model:'):
                        modulation = line.split(':', 1)[1].strip()
                    if line.startswith('% Date:'):
                        date = line.split(':', 1)[1].strip()

        # Extract parameter sweep
        param_value_pairs = re.findall(r'(\b(?!t\b)\w+)=([0-9.eE+-]+)', header_line)

        if param_value_pairs:
            sweep_names, values = zip(*param_value_pairs)
            sweep_set = set(sweep_names)

            if len(sweep_set) == 1:
                sweep = sweep_names[0]
                parametric_sweep = sorted(set(map(float, values)))
                studies = len(parametric_sweep)
            else:
                raise ValueError(f"Multiple sweep parameters found: {sweep_set}")
        else:
            raise ValueError("No sweeping parameter found in header.")

        is_cartesian = ("x" in header_line) and ("y" in header_line)
        # Clean up header
        if is_cartesian:
            header_line_mod = re.sub(r'(?<=\s)x(?=\s)', r'x/*/', header_line)
            header_line_mod = re.sub(r'(?<=\s)y(?=\s)', r'y/*/', header_line_mod)
        else:
            header_line_mod = re.sub(r'(?<=\s)r(?=\s)', r'r/*/', header_line)
            header_line_mod = re.sub(r'(?<=\s)z(?=\s)', r'z/*/', header_line_mod)
        pattern = fr'{sweep}=([\d.eE+-]+)'
        header_line_mod = re.sub(pattern, r'\g<0>/*/', header_line_mod)

        columns = [col.strip() for col in header_line_mod.split('/*/') if col.strip()]

        # Load data
        rough_file = pd.read_csv(file, skiprows=9, header=None, delim_whitespace=True)
        rough_file.columns = columns

        # Cache r and z
        if is_cartesian:
            globals()['r'] = identify_variable("% x")  * 1e3
            globals()['z'] = identify_variable("y") * 1e3
        else:
            globals()['r'] = identify_variable("% r")
            globals()['z'] = identify_variable("z")

        return sweep, parametric_sweep, studies, header_line, modulation, date, is_cartesian, criterion


def identify_variable(variable):
    values = [col for col in columns if col.startswith(variable)]
    sub_array = rough_file[values]
    return sub_array


def get_times_from_variable(variable):
    times_by_parameter = {amp: [] for amp in parametric_sweep}
    time_amp_pattern = re.compile(r"t=([\d.]+),\s*" + re.escape(sweep) + r"=([\d.eE+-]+)")
    sub_array = identify_variable(variable)
    for col in sub_array.columns:
        match = time_amp_pattern.search(col)
        if match:
            t_val = float(match.group(1))
            param_val = float(match.group(2))
            if param_val in times_by_parameter:
                times_by_parameter[param_val].append(t_val)
    for param in times_by_parameter:
        times_by_parameter[param].sort()
    return np.array([times_by_parameter[param] for param in parametric_sweep])


def build_data_array(variable):
    T_array = identify_variable(variable)
    time_array = get_times_from_variable(variable)

    n_sweep = len(parametric_sweep)
    n_time = time_array.shape[1]
    n_points = r.shape[0]

    result = np.zeros((n_sweep, n_time, n_points, 4))
    time_amp_pattern = re.compile(r"t=([\d.]+),\s*" + re.escape(sweep) + r"=([\d.eE+-]+)")
    T_column_map = {}
    for col in T_array.columns:
        match = time_amp_pattern.search(col)
        if match:
            t_val = float(match.group(1))
            amp_val = float(match.group(2))
            T_column_map[(amp_val, t_val)] = col

    for i, amp in enumerate(parametric_sweep):
        for j, t in enumerate(time_array[i]):
            col = T_column_map.get((amp, t))
            if col:
                result[i, j, :, 0] = t
                result[i, j, :, 1] = r.values.flatten()
                result[i, j, :, 2] = z.values.flatten()
                result[i, j, :, 3] = T_array[col].values.flatten()
            else:
                raise ValueError(f"Missing column for t={t}, {sweep}={amp}")
    return result


def create_time_sequence(r_query, z_query, variable, tol=1e-6):

    if criterion:
        result = []
        times = rough_file[:, 0]
        for i, h in enumerate(header_line):
            if h.startswith(variable):
                temps = rough_file[:, i]
                result.append(np.column_stack((times, temps)))        
        return result
    
    else:
        data_array = build_data_array(variable)

        n_sweep, n_time, n_points, _ = data_array.shape
        r_all = data_array[0, 0, :, 1]
        z_all = data_array[0, 0, :, 2]

        distances = np.sqrt((r_all - r_query)**2 + (z_all - z_query)**2)
        k = np.argmin(distances)

        if distances[k] > tol:
            print(f"Warning: Closest point {r_all[k], z_all[k]} is {distances[k]:.3e} away from requested (r,z)")

        result = []
        for i in range(n_sweep):
            times = data_array[i, :, k, 0]
            temps = data_array[i, :, k, 3]
            result.append(np.column_stack((times, temps)))

        return result

def cut_time_sequence(data,start,end):
    data = data[data[:, 0] >= start]
    data = data[data[:, 0] <= end]
    data[:, 0] = data[:, 0] - start
    return data

def fun_linear(x, a, b):
    return a * x + b

def fun_cuadratic(x, a, b, c):
    return a * x**2 + b * x + c

def fun_sin(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def linear_fit(x, y):
    popt, _ = curve_fit(fun_linear, x, y)
    a = popt[0]
    y_fit = fun_linear(x, *popt)
    return y_fit,a  # shape matches y

def cuadratic_fit(x, y):
    popt, _ = curve_fit(fun_cuadratic, x, y)
    y_fit = fun_cuadratic(x, *popt)
    return y_fit  # shape matches y

def plot_linear_fit(x, y):
    y_fit = linear_fit(x, y)
    plt.plot(x, y, 'o', label="Original")
    plt.plot(x, y_fit, '-', label="Linear fit")
    plt.legend()
    plt.show()

def detrend(x, y):
    y_lin, a = linear_fit(x, y)
    y_delinear = y - y_lin
    print(a)

    y_cua = cuadratic_fit(x, y_delinear)
    y_detrended = y_delinear - y_cua
    return y_detrended

def detrend_linear(x, y):
    y_lin, a = linear_fit(x, y)
    y_delinear = y - y_lin
    return y_delinear

def moving_mean_detrend(x, y, window_seconds):
    t = np.asarray(x)
    v = np.asarray(y)
    n = len(t)
    bg = np.empty(n)
    half = window_seconds / 2.0

    for i in range(n):
        left = np.searchsorted(t, t[i] - half, 'left')
        right = np.searchsorted(t, t[i] + half, 'right')
        bg[i] = v[left:right].mean()

    y_detrended = v - bg
    i_start = np.searchsorted(t, t[0] + half, 'left')
    i_end = np.searchsorted(t, t[-1] - half, 'right')

    return t[i_start:i_end], y_detrended[i_start:i_end]

def sinusoidal_fit(x, y, f0):
    initial_guess = [max(y) - min(y), 2 * np.pi * f0, 0, np.mean(y)]
    popt, pcov = curve_fit(fun_sin, x, y, p0=initial_guess)

    a, b, c, d = popt
    y_fit = fun_sin(x, a, b, c, d)

    amplitude = abs(popt[0])
    phase_shift = popt[2]

    return y_fit, amplitude, phase_shift

def amplitude_procedure(variable,f0, start_time, end_time, r, z, moving_mean=True, show=0):
    sequence = create_time_sequence(r, z, variable)
    detrend_sequence = []
    amplitudes = np.zeros(studies)

    for i in range(studies):
        sequence[i] = cut_time_sequence(sequence[i], start_time, end_time)
        times = sequence[i][:, 0]
        temps = sequence[i][:, 1]

        if moving_mean:
            times, detrended_data = moving_mean_detrend(times, temps, 1/f0[i])
        else:
            detrended_data = detrend(times, temps)
        detrend_sequence.append([times, detrended_data])

        y_fit, amplitude, phase_shift = sinusoidal_fit(times, detrended_data, f0[i])
        amplitudes[i] = amplitude

        if show:
            fig, ax = plt.subplots(figsize=(11, 8), facecolor='white')
            ax.plot(times, y_fit, label="Fit", color="red")
            ax.plot(times, detrended_data, ".", label="Detrended", color="black")

            # Relative position (0.75 means 75% to the right, 0.62 means 62% up)
            plt.figtext(0.3, 0.02,  # (x, y) in figure coordinates — 0.1 from left, -0.05 below the plot
                        f"Amplitude = {round(amplitude, 6)}",
                        ha='left', va='top', fontsize=13, color="blue")

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Quantity')

            ax.legend()
            plt.show()

    return amplitudes, detrend_sequence

def cross_correlation_analysis(data1, data2, f0, show=0):
    time1 = data1[0]
    time2 = data2[0]
    signal1 = data1[1] / max(data1[1])
    signal2 = data2[1] / max(data2[1])

    # Interpolation factor (e.g., 5x finer)
    T = 1/f0
    points = T/0.1
    step_old = time1[1] - time1[0]
    factor = int(points/step_old)
    new_time = np.linspace(time1[0], time1[-1], len(time1) * factor)

    # Cubic spline interpolation
    cs1 = CubicSpline(time1, signal1)
    cs2 = CubicSpline(time2, signal2)

    interp_signal1 = cs1(new_time)
    interp_signal2 = cs2(new_time)

    # Updated time step and sampling rate
    step = new_time[1] - new_time[0]
    sampling_rate = 1 / step

    # Cross-correlation of interpolated signals
    correlation = correlate(interp_signal1, interp_signal2, mode='full')
    center = len(correlation) // 2

    # Time lag and phase shift
    lag = np.argmax(correlation) - center
    time_delay = lag / sampling_rate
    phase_shift = 2 * np.pi * f0 * time_delay

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), facecolor='white')
    plt.rcParams.update({'font.size': 14})

    if show:
        # Plot original signals
        axs[0].plot(time1, signal1, color="blue")
        axs[0].plot(time2, signal2, color="black")
        #axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Normalized quantities')
        axs[0].set_title(f"Original signals", fontsize=15)

        # Cross-correlation plot
        lags = np.arange(-center, center + 1)
        axs[1].plot(lags, correlation, color="red")
        axs[1].axvline(x=lag, color='black', linestyle='--', label="Detected maximum")
        axs[1].set_title('Cross-correlation function', fontsize=15)
        axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.85))
        axs[1].set_xlabel('Lag')
        axs[1].set_ylabel('Cross-correlation')

        # Plot shifted signals
        axs[2].plot(time1, signal1, label=f"shifted", color="blue")
        axs[2].plot(time2 + time_delay, signal2,label="original", color="black")
        #axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Normalized quantities')
        axs[2].set_title('Shifted signals', fontsize=15)

        # Annotations
        #fig.text(0.75, 0.62, f"Time delay = {round(time_delay, 4)} s", ha='left', va='top', color="blue", fontsize=13)
        #fig.text(0.75, 0.59, f"Time step = {round(step, 5)} s", ha='left', va='top', color="blue", fontsize=13)
        #fig.text(0.75, 0.56, f"Phase shift = {round(phase_shift, 4)} rad", ha='left', va='top', color="blue", fontsize=13)

        plt.tight_layout()
        #plt.savefig("crosscorelation.png", dpi=300, bbox_inches="tight")
        plt.show()

    return time_delay, phase_shift


def cross_corelation_procedure(variable,top, f0, show=0):
    time_delay = np.zeros(studies)
    phase_shift = np.zeros(studies)
    for i in range(studies):
        time_delay[i], phase_shift[i] = cross_correlation_analysis(variable[i], top[i], f0[i],show)
    return time_delay, phase_shift


