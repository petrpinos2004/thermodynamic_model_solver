import python_core.SULTAN_helpers as SULTAN
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

class Data():

    def __init__(self, name, f0 = [0.01,0.01,0.01], plot=False):

        #-------------------------- Starting sequence to guess looping parameters, names and dates--------------------

        input_filename = Path(
            rf'C:\Users\sulta\Documents\UPT  - kryogenika\starsi_veci_bakalar\thermodynamic_model\data_output\raw_simulation_data\{name}.txt'
        )
        output_filename = Path(
            rf'C:\Users\sulta\Documents\UPT  - kryogenika\starsi_veci_bakalar\thermodynamic_model\data_output\processed_simulation_data\{name}_output.csv'
        )

        sweep, parametric_sweep, studies, header,modulation, date, is_cartesian, criterion = SULTAN.starting_procedure(input_filename)
        print(modulation)

        print(f"Sweeping parameter identified as: {sweep} \n "
            f"with values {parametric_sweep} \n")

        if not len(f0)==len(parametric_sweep):
            f0 = f0 * len(parametric_sweep)

        #-------------------------- Data processing procedure using defined database and functions over it------------

        if is_cartesian:
            #Cartesian
            top_plate = [0,150]
            bottom_plate = [0,-150]
            bulk_point = [0,0]
            cut_time = [1/f0[0],5000]
        else:
            #Radial
            top_plate = [0,300]
            bottom_plate = [0,0]
            bulk_point = [0,100]
            cut_time = [50,400]

        #--------------------------Amplitudes--------------------------------------------------------

        if criterion:
            Tt_amplitude, detrend_sequence_top = SULTAN.amplitude_procedure("Ref", f0, cut_time[0],cut_time[1], top_plate[0], top_plate[1], show=plot)
            Tb_amplitude, detrend_sequence_bottom = SULTAN.amplitude_procedure("Tb",f0, cut_time[0],cut_time[1], bottom_plate[0], bottom_plate[1], show=plot)
            Tp_amplitude, detrend_sequence_bulk = SULTAN.amplitude_procedure("Age", f0, cut_time[0],cut_time[1], top_plate[0], top_plate[1], show=plot)
        else:
            #Top plate
            Tt_amplitude, detrend_sequence_top = SULTAN.amplitude_procedure("T", f0, cut_time[0],cut_time[1], top_plate[0], top_plate[1], show=plot)

            #Temperatures
            Tb_amplitude, detrend_sequence_bottom = SULTAN.amplitude_procedure("T",f0, cut_time[0],cut_time[1], bottom_plate[0], bottom_plate[1], show=plot)
            Tp_amplitude, detrend_sequence_bulk = SULTAN.amplitude_procedure("T",f0, cut_time[0],cut_time[1], bulk_point[0], bulk_point[1], show=plot)

            #Pressure
            p_amplitude, detrend_sequence_p = SULTAN.amplitude_procedure("p",f0, cut_time[0],cut_time[1], bulk_point[0], bulk_point[1], show=plot)

            #Heat across the boundary
            Q_amplitude, detrend_sequence_Q = SULTAN.amplitude_procedure("Q",f0, cut_time[0],cut_time[1], bulk_point[0], bulk_point[1], show=plot)
            #Power across the boundary
            P_amplitude, detrend_sequence_P = SULTAN.amplitude_procedure("P",f0, cut_time[0],cut_time[1], bulk_point[0], bulk_point[1], show=plot)
            #internal energy change
            U_amplitude, detrend_sequence_U = SULTAN.amplitude_procedure("U_V",f0, cut_time[0],cut_time[1], bulk_point[0], bulk_point[1], show=plot)

        def modulation_depth(x):
            return x/Tt_amplitude

        x1_list = modulation_depth(Tb_amplitude)
        x2_list = modulation_depth(Tp_amplitude)

        #--------------------------Delay - crosscorelation------------------------------------------------------------

        if criterion:
            #Temperatures
            time_delay_Tb, phase_shift_Tb = SULTAN.cross_corelation_procedure(detrend_sequence_bottom,detrend_sequence_top, f0, plot)
            time_delay_Tp, phase_shift_Tp = SULTAN.cross_corelation_procedure(detrend_sequence_bulk,detrend_sequence_top, f0, plot)
        else:
            #Temperatures
            time_delay_Tb, phase_shift_Tb = SULTAN.cross_corelation_procedure(detrend_sequence_bottom,detrend_sequence_top, f0, plot)
            time_delay_Tp, phase_shift_Tp = SULTAN.cross_corelation_procedure(detrend_sequence_bulk,detrend_sequence_top, f0, plot)

            #pressure
            time_delay_p, phase_shift_p = SULTAN.cross_corelation_procedure(detrend_sequence_p,detrend_sequence_top, f0, plot)

            #Q
            #Q je kvuli vypoctu v comsolu s minusem (smer heat fluxu), je tak potreba rucne otocit znaminko
            for i in range(studies):
                detrend_sequence_Q[i][1] = -detrend_sequence_Q[i][1]
            time_delay_Q, phase_shift_Q = SULTAN.cross_corelation_procedure(detrend_sequence_Q,detrend_sequence_top, f0, plot)

            #U
            time_delay_U, phase_shift_U = SULTAN.cross_corelation_procedure(detrend_sequence_U,detrend_sequence_top, f0, plot)

            #for i in range(studies):
            #   detrend_sequence_P[i][1] = -detrend_sequence_P[i][1]
            time_delay_P, phase_shift_P = SULTAN.cross_corelation_procedure(detrend_sequence_P,detrend_sequence_top, f0, plot)

        #-------------------------Save result to .csv file-----------------------------------------------------------

        with open(output_filename, "w") as f:

            if criterion:
                f.write(f"Modulation\tDate\tsweep through {sweep}\tf [Hz]\tAT [K]\tAB [K]\tAGe [K]\tAGe/AT\tAB/AT\ttau_TB [s]\ttau_TGe [s]\n")
                
                for sweep_val, f_val, tta, tba, tpa, x1, x2, x3, x4 in zip(
                    parametric_sweep,
                    f0,
                    Tt_amplitude,
                    Tb_amplitude,
                    Tp_amplitude,
                    x1_list,
                    x2_list,
                    time_delay_Tb,
                    time_delay_Tp,
                ):
                
                    f.write(
                        f"{modulation}\t{date}\t{sweep_val}\t{f_val}\t{tta}\t{tba}\t{tpa}\t{x2}\t{x1}\t{x3}\t{x4}\n"
                    )
            else:
                f.write(f"Modulation\tDate\tsweep through {sweep}\tf [Hz]\tAT [K]\tAB [K]\tAGe [K]\tAGe/AT\tAB/AT\tAp [Pa]\tAQ "
                        f"[J]\tAU [J]\tAP (vykon)\ttau_TB [s]\ttau_TGe [s]\ttau_p [s]\ttau_Q [s]\ttau_U [s]\t tau_P\n")

                for sweep_val, f_val, tta, tba, tpa, x1, x2, x3, x4, x5, u1, u2, u3, u4, u5, u6, u7 in zip(
                    parametric_sweep,
                    f0,
                    Tt_amplitude,
                    Tb_amplitude,
                    Tp_amplitude,
                    x1_list,
                    x2_list,
                    p_amplitude,
                    Q_amplitude,
                    U_amplitude,
                    P_amplitude,
                    time_delay_Tb,
                    time_delay_Tp,
                    time_delay_p,
                    time_delay_Q,
                    time_delay_U,
                    time_delay_P
                ):

                    f.write(
                        f"{modulation}\t{date}\t{sweep_val}\t{f_val}\t{tta}\t{tba}\t{tpa}\t{x2}\t{x1}\t{x3}\t{x4}\t{x5}\t{u1}\t{u2}"
                        f"\t{u3}\t{u4}\t{u5}\t{u6}\t{u7}\n"
                    )

        print(f'Data succesfully saved to {output_filename}')

#sequence = SULTAN.create_time_sequence(bulk_point[0], bulk_point[1], "T")

#data = sequence[2]
#data = SULTAN.cut_time_sequence(data,200,2000)
#x = data[:,0]
#y = data[:,1]
#new = SULTAN.detrend(x,y)

#plt.figure(figsize=(5, 3.5), facecolor="white", dpi=300)
#plt.rcParams.update({'font.size': 12})

#plt.plot(x, new,".", color="blue")
#plt.xlabel("Time [s]")
#plt.ylabel("Temperature [K]")
#plt.savefig("decuadratic sequence.png", dpi=300, bbox_inches="tight")
#plt.show()