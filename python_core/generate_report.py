from pathlib import Path
import pandas as pd
from fractions import Fraction
from thermodynamic_model import thermodynamic_model

# Initialize physics dict to extract the values populated inside __init__
physics = {}
maths = {}
_ = thermodynamic_model(physics=physics, maths=maths, name="report")
gamma = physics.get('ratio')
xi = physics['xi']

frac = Fraction(gamma).limit_denominator(100)
gamma_str = f"{frac.numerator}d{frac.denominator}"
xi_str = str(xi).replace(".", "p")
name = f"gamma_{gamma_str}_xi_{xi_str}_bulk"

# --- YOUR ORIGINAL LOGIC UNCHANGED BELOW ---
input_filename = Path('data_output') / 'processed_simulation_data'
output_filename = Path('data_output') / 'results' / f'{name}_output.csv'

csv_files = sorted(input_filename.glob("*_output.csv"))
all_dfs = []

for file_path in csv_files:
    df = pd.read_csv(file_path, sep=None, engine="python")
    df.insert(0, "Source_File", file_path.stem)
    all_dfs.append(df)

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_filename, sep="\t", index=False)

    print(f"Successfully merged {len(csv_files)} files into:\n{output_filename.resolve()}")
else:
    print("No matching CSV files found.")