import numpy as np
import pandas as pd
import subprocess
from utils import get_error, extrac_column_info
import os
import datetime
import accelerate
import timeit
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

exe_path = os.path.join(PROJECT_ROOT, config["paths"]["exe_path"])
data_path = os.path.join(PROJECT_ROOT, config["paths"]["data_path"])
working_dir = os.path.join(PROJECT_ROOT, config["paths"]["working_dir"])
link_perform = os.path.join(PROJECT_ROOT, config["paths"]["link_performance"])
link_perform_odlink = os.path.join(PROJECT_ROOT, config["paths"]["link_perform_odlink"])
initial_demand_xlsm = os.path.join(PROJECT_ROOT, config["paths"]["initial_demand_xlsm"])
results_path = os.path.join(PROJECT_ROOT, "results")
initial_demand_xlsm = os.path.join(PROJECT_ROOT, config["paths"]["initial_demand_xlsm"])

experience_mode = 'column_experi' # ["record_best", "column_experi", "origin_data"]
np.set_printoptions(threshold=np.inf)

# simulate 
# Run the executable
subprocess.run(["wine64", exe_path], cwd=data_path)

if experience_mode == "column_experi":
    file_path = initial_demand_xlsm
    df_ini = extrac_column_info(file_path)
    matrix_holder = []
    # over write the new content, avoid diagonal keep as 0, avoid first row and column
    for i, row in enumerate (df_ini.index[0:], start=0):
        for j, col in enumerate(df_ini.columns[0:], start=0):
            if i != j:
                # Random value assignment, can be adjusted if needed
                matrix_holder.append(df_ini.at[row, col])
            else:
                pass
                # print("pass: row: {}, col: {}".format(row, col))
    starting_solution = np.array(matrix_holder)#(3080,)

starting_solution = starting_solution.reshape(1, -1) # (1, 3080)

# function to construct 56x56 matrix from OD vector with diagonal as 0s
def unflatten_od_vector(od_vector, size = 56):
    matrix = np.zeros((size, size))
    k = 0
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i, j] = od_vector[k]
                k+=1
            else:
                pass
    return matrix

flat_vector = starting_solution[0]  # shape (3080,)
initial_matrix = unflatten_od_vector(flat_vector, size=56)
baseline_matrix = initial_matrix.copy()

# flatten a 56x56 matrix (ignoring diagonals, i != j) into a 3080-element array.
def flatten_od_matrix(matrix, size=56):
    holder = []
    for i in range(size):
        for j in range(size):
            if i != j:
                holder.append(matrix[i, j])
    return np.array(holder)  # shape (3080,)

# get mse and volume
def get_mse_volume(matrix, data_path, link_perform): 
        # matrix: 56 * 56
        # Convert matrix to demand file and run simulation (You need to implement this)
        index_and_columns = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '602', '615', '617', 
        '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '633', '637', '640', '645', '646', '647', 
        '649', '650', '651', '652', '653', '654', '766', '767', '2061', '2125', '2136', '2137', '2142', '2146', '2147', 
        '2148', '2166', '2197'
        ]

        # Initialize the DataFrame.
        df = pd.DataFrame(index=index_and_columns, columns=index_and_columns)

        # over write the new content, avoid diagonal keep as 0, avoid first row and column
        for i, row in enumerate (df.index[0:], start=0):
            for j, col in enumerate(df.columns[0:], start=0):
                # if i != j-1:
                    # Random value assignment, can be adjusted if needed
                    df.at[row, col] = matrix[i, j]

        demand_path = os.path.join(data_path, "demand.csv")
        df.to_csv(demand_path)
        # simulate 
        # Run the executable
        subprocess.run(["wine64", exe_path], cwd=data_path)
        mse = get_error(file_path=link_perform)
        link_df = pd.read_csv(link_perform)
        
        # gather volume in a dict
        volume = {}
        if 'link_id' in link_df.columns and 'volume' in link_df.columns:
            for _, row in link_df.iterrows():
                link_id = str(row['link_id']).strip()
                volume[link_id] = float(row['volume'])
        else:
            print('Error: link_id or volume not found in link_performance.csv')

        return mse, volume


# Returns a copy of matrix where (i,j) entry is increased by delta. Ensures no negative values.
def change_od_element(matrix, i, j, delta = 10.0):
    new_matrix = matrix.copy()
    val = matrix[i,j] + delta
    new_matrix[i,j] = max(0.0, val)
    return new_matrix

 
def main():
    baseline_mse, baseline_volume = get_mse_volume(baseline_matrix, data_path, link_perform)
    print(f"Baseline MSE: {baseline_mse:.4f}")

    baseline_df = pd.read_csv(link_perform).copy()
        
    # Build a dictionary: link_id -> set of (i,j) pairs
    link_to_odpairs = {lid: set() for lid in baseline_volume.keys()}

    size = 56
    delta_change = 2.0 # delta = how much are we increasing each element by
    epsilon = 1e-3  # threshold to decide if a link is changed

    for i in range(size):
        for j in range(size):
            # skip diagonal
            if i == j:
                continue

            test_matrix = change_od_element(baseline_matrix.copy(), i, j, delta_change)
            _, test_volume = get_mse_volume(test_matrix, data_path, link_perform)

            for link_id in baseline_volume:
                old_count = baseline_volume[link_id]
                new_count = test_volume.get(link_id, 0.0)
                if abs(new_count - old_count) > epsilon:
                    link_to_odpairs[link_id].add((i, j))

    od_pairs_list = []
    for _, row in baseline_df.iterrows():
        lid = str(row['link_id'])
        # gather the set of pairs that changed the link
        od_pairs = link_to_odpairs.get(lid, set())
        if od_pairs:
            pairs_str = ", ".join(f"({p[0]},{p[1]})" for p in sorted(od_pairs))
        else:
            pairs_str = ""
        od_pairs_list.append(pairs_str)
    
    baseline_df['od_pairs'] = od_pairs_list
    baseline_df.to_csv(link_perform_odlink, index=False)
    print(f"Created {link_perform_odlink} with OD pair mappings.")

if __name__ == "__main__":
    main()