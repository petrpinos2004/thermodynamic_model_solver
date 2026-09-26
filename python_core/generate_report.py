from pathlib import Path
import pandas as pd

print("Please add simulation name to save in /data_output/results folder:")
print("------------------------------------------------------------------")
name = input()
print()

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