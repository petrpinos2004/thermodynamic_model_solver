#!/bin/bash

# Parse xi and gamma arrays from command line arguments
for arg in "$@"; do
    case $arg in
        xi=*)
            raw="${arg#xi=[}"
            raw="${raw%]}"
            IFS=',' read -r -a xi_values <<< "$raw"
            ;;
        gamma=*)
            raw="${arg#gamma=[}"
            raw="${raw%]}"
            IFS=',' read -r -a gamma_values <<< "$raw"
            ;;
    esac
done

# Set defaults if none provided
if [ ${#xi_values[@]} -eq 0 ]; then xi_values=('None'); fi
	if [ ${#gamma_values[@]} -eq 0 ]; then gamma_values=('None'); fi

# Sweep over all combinations
for xi in "${xi_values[@]}"; do
    for gamma in "${gamma_values[@]}"; do
        echo "Running sweep iteration: xi=$xi, gamma=$gamma"

        python h328.py "$xi" "$gamma" 0 &
        python h429.py "$xi" "$gamma" 1 &
        python h430.py "$xi" "$gamma" 2 &
        python h431.py "$xi" "$gamma" 3 &
        python h432.py "$xi" "$gamma" 4 &
        python h433.py "$xi" "$gamma" 5 &
        wait

        python python_core/generate_report.py "$xi" "$gamma"
    done
done
