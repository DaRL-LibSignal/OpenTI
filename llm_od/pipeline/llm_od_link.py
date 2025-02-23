import numpy as np
import pandas as pd
import subprocess
from utils import get_error, extrac_column_info
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import timeit
import random
import os
import datetime
import accelerate
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import re
import json

# configure env keys
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

experience_mode = 'column_experi' # ["record_best", "column_experi", "origin_data"]
np.set_printoptions(threshold=np.inf)

# load directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# define the path to the executable
exe_path = os.path.join(PROJECT_ROOT, config["paths"]["exe_path"])
data_path = os.path.join(PROJECT_ROOT, config["paths"]["data_path"])
working_dir = os.path.join(PROJECT_ROOT, config["paths"]["working_dir"])
link_performance_csv = os.path.join(PROJECT_ROOT, config["paths"]["link_performance"])
link_perform_odlink = os.path.join(PROJECT_ROOT, config["paths"]["link_perform_odlink"])
initial_demand_xlsm = os.path.join(PROJECT_ROOT, config["paths"]["initial_demand_xlsm"])
results_path = os.path.join(PROJECT_ROOT, "results")
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

# Run the simulation executable
subprocess.run(["wine64", exe_path], cwd=data_path)

if experience_mode == "column_experi":
    file_path  = initial_demand_xlsm
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

# flatten a 56x56 matrix (ignoring diagonals, i != j) into a 3080-element array.
def flatten_od_matrix(matrix, size=56):
    holder = []
    for i in range(size):
        for j in range(size):
            if i != j:
                holder.append(matrix[i, j])
    return np.array(holder)  # shape (3080,)

def calculate_mse(matrix, data_path, exe_path, results_path, link_performance_csv): 
        # matrix: 56 * 56
        # Convert matrix to demand file and run simulation

        index_and_columns = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '602', '615', '617', 
        '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '633', '637', '640', '645', '646', '647', 
        '649', '650', '651', '652', '653', '654', '766', '767', '2061', '2125', '2136', '2137', '2142', '2146', '2147', 
        '2148', '2166', '2197'
        ]

        # Initialize the DataFrame.
        df = pd.DataFrame(index=index_and_columns, columns=index_and_columns)

        # overwrite the new content, avoid diagonal keep as 0, avoid first row and column
        for i, row in enumerate (df.index[0:], start=0):
            for j, col in enumerate(df.columns[0:], start=0):
                # if i != j-1:
                    # Random value assignment, can be adjusted if needed
                    df.at[row, col] = matrix[i, j]

        demand_path = os.path.join(data_path, "demand.csv")
        df.to_csv(demand_path)
        # simulate 
        # Run the executable
        start = timeit.default_timer()
        subprocess.run(["wine64", exe_path], cwd=data_path)
        stop = timeit.default_timer()
        total = stop - start
        print(f"Time taken for one simulation: {total}")

        mse_start = timeit.default_timer()
        mse = get_error(file_path=link_performance_csv)
        mse_stop = timeit.default_timer()
        mse_total = mse_stop - mse_start
        print(f"Time taken for MSE: {mse_total}")
        mse_history_file = os.path.join(results_path, "mse_history.txt")
        with open(mse_history_file, 'a+') as file:
            file.write(str(mse)+"\n")
        return mse

# returns a dict with abs errors of all links
def calculate_abs_error(link_performance_csv):
    link_df = pd.read_csv(link_performance_csv)
    abs_error_dict = {}
    for _, row in link_df.iterrows():
        link_id = str(row['link_id']).strip()
        if pd.isna(row['volume']) or pd.isna(row['obs_count']):
            continue
        
        volume = float(row['volume'])
        obs_count = float(row['obs_count'])
        abs_error = abs(volume - obs_count)
        abs_error_dict[link_id] = abs_error
    # sort dict in decreasing order of abs_error values 
    sorted_abs_error_dict = sorted(abs_error_dict, key = abs_error_dict.get, reverse = True)
    return sorted_abs_error_dict

# returns abs error for a particular link id
def get_abs_error(link_id, link_performance_csv):
    link_df = pd.read_csv(link_performance_csv)
    row = link_df[link_df['link_id'].astype(str) == str(link_id)]
    if row.empty:
        print(f"Warning: No entry found for link_id={link_id}")
        return float('inf')
    
    # extract volume and observed count
    volume = float(row.iloc[0]['volume'])
    obs_count = float(row.iloc[0]['obs_count'])

    # compute absolute error
    abs_error = abs(volume - obs_count)
    return abs_error

# returns simulated_volume and obs_count (ground truth) for the given link_id
def get_link_data(link_id, link_performance_csv):
    df = pd.read_csv(link_performance_csv)
    row = df[df['link_id'].astype(str) == str(link_id)]
    if row.empty:
        print(f"Warning: No entry found for link_id={link_id} in {link_performance_csv}")
        return None, None
    # Check for NaN
    if pd.isna(row.iloc[0]['volume']) or pd.isna(row.iloc[0]['obs_count']):
        print(f"Warning: Link {link_id} has blank volume or obs_count, skipping.")
        return None, None
    volume = float(row.iloc[0]['volume'])
    obs_count = float(row.iloc[0]['obs_count'])
    return volume, obs_count

# sample od pairs link_id and path to the link_performance_odlink_delta2.csv
def sample_od_pairs(link_id, link_perform_odlink, current_matrix):
    link_df = pd.read_csv(link_perform_odlink)
    row = link_df[link_df['link_id'] == link_id]
    if row.empty:
        # if no row is found, return an empty list
        print(f"No entry found for link_id={link_id}")
        return []
    # get od pairs from row
    od_str = str(row.iloc[0]['od_pairs']).strip()
    if not od_str:
        print(f"No OD pairs listed for link_id={link_id}")
        return []
    # parse string for pairs of form (i, j)
    pairs_list = re.findall(r"\(\d+,\d+\)", od_str)
    sample_size = min(20, len(pairs_list))
    sampled_od_pairs = random.sample(pairs_list, sample_size)
    # i,j and value
    od_pair_val = []
    for rp in sampled_od_pairs:
        i_str, j_str = rp.strip("()").split(",")
        i_int, j_int = int(i_str), int(j_str)
        flow_val = current_matrix[i_int, j_int]
        od_pair_val.append(((i_int, j_int), flow_val))
    return od_pair_val

# load LLaMa 3.3-70B model
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_auth_token=HUGGING_FACE_API_KEY
)
# Initialize an empty model
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16)
model.eval()

def model_prompt(link_id, abs_error, sampled_od_pairs, simulated_vol, obs_count):
    """
    sampled_od_pairs is now a list of [((i_int, j_int), flow_val), ...]
    """
    pairs_str = ""
    for (i_int, j_int), flow_val in sampled_od_pairs:
        pairs_str += f"({i_int},{j_int}), {flow_val:.2f}\n"
    sample_size = len(sampled_od_pairs)
    prompt = f"""
    System Description:
    We are calibrating a 56x56 Origin-Destination (OD) matrix for a transportation network. Each entry [i, j] represents the number of trips from origin i to destination j. 
    This OD matrix is used in a simulation to generate traffic volume counts on various links in the network.

    You will be provided with the following details of one link:
    Link ID: {link_id}
    The simulated volume: {simulated_vol}
    The ground truth volume: {obs_count}
    The absolute error, which is calculated as: abs(Simulated Volume - Ground Truth Volume) = {abs_error}

    The following {sample_size} randomly sampled OD elements (i,j) that contribute to this link, along with their current flow values:
    {pairs_str}

    Your Task:
    Adjust ONLY these OD elements' flow values to reduce the absolute error, thereby improving the alignment between the simulation results and real-world traffic observations.

    Response Constraints and Format:
    - Do not return any placeholder text.
    - Return ONLY the updated values of the {sample_size} OD elements with their indices, one per line, in the format:
      [(i, j), new_value]
      [(i2, j2), new_value_2]
      ...

    """
    return prompt

def parse_llm_output(model_output):
    # assuming model output is in the form
    lines = model_output.strip().split("\n")
    parsed_data = {}

    pattern = re.compile(r'^\[\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*([\d\.]+)\]$')
    # iterate through all lines to parse through output
    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            i_str, j_str, val_str = match.groups()
            i = int(i_str)
            j = int(j_str)
            new_value = float(val_str)  # parse as float in case of decimals
            parsed_data[(i, j)] = new_value
        else:
            print(f"Skipping line that doesn't match format: {line}")
            pass
    return parsed_data

def generate_output(prompt, model, tokenizer, working_dir, link_id, attempt, results_path):
    matrix_start = timeit.default_timer()
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 4000,
            temperature = 0.7,
            do_sample = True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logs_path = os.path.join(results_path, "logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    output_path = os.path.join(logs_path, f"llama_raw_output{link_id}_attempt{attempt}.txt")
    with open(output_path, "w+") as f:
        f.write(generated_text)
    # parse llm output
    output_od_pairs = parse_llm_output(generated_text)
    matrix_stop = timeit.default_timer()
    total_matrix = matrix_stop - matrix_start
    print(f"Time taken to generate output: {total_matrix}")
    return output_od_pairs

def update_od_matrix(od_matrix, updated_pairs):
    new_matrix = od_matrix.copy()
    for (i,j), val in updated_pairs.items():
        if i < 0 or i >= 56 or j < 0 or j >= 56:
            print(f"Warning: LLM suggested an out-of-bounds index ({i}, {j}). Skipping this update.")
            continue
        new_matrix[i][j] = max(0.0, val)
    return new_matrix

def log_improvement_and_save(
    results_path, 
    global_iter, 
    link_id, 
    attempt, 
    old_mse, 
    new_mse, 
    updated_pairs, 
    best_matrix
):
    # 1. write detailed improvement log
    log_path = os.path.join(results_path, "detailed_log.txt")
    with open(log_path, "a+") as log_file:
        log_file.write("=== SUCCESSFUL ITERATION ===\n")
        log_file.write(f"Global Iteration: {global_iter}\n")
        log_file.write(f"Link: {link_id}\n")
        log_file.write(f"Attempt: {attempt+1}\n")  # Add +1 if attempt is zero-based
        log_file.write(f"Old MSE: {old_mse:.4f}, New MSE: {new_mse:.4f}\n")
        log_file.write("Updated OD Pairs (i,j -> new_value):\n")
        for (i, j), val in updated_pairs.items():
            log_file.write(f"  ({i},{j}) -> {val:.2f}\n")
        log_file.write("\n")

    # save current best matrix
    matrix_path = os.path.join(results_path, "best_matrix_current.csv")
    with open(matrix_path, "w") as csv_file:
        for row in best_matrix:
            row_str = ",".join(str(int(x)) for x in row)
            csv_file.write(row_str + "\n")
    print(f"Saved best_matrix to {matrix_path}")

# LLM Optimization Pipeline
baseline_matrix = initial_matrix.copy()
baseline_mse = calculate_mse(baseline_matrix, data_path, exe_path, results_path, link_performance_csv)
best_mse = baseline_mse
best_matrix = baseline_matrix.copy()
print(f"Baseline MSE: {baseline_mse}")
sorted_links = calculate_abs_error(link_performance_csv)
top_links = sorted_links[:20] # top 20 links
print(f"Top 20 links with highest abs error: {top_links}")
current_matrix = baseline_matrix.copy()

num_iterations = config["hyperparams"]["num_iterations"] # number of attempts per link, currently 5
# in each global iteration, if a link has failed to optimize we add it here to this dictionary
max_global_iterations = config["hyperparams"]["max_global_iterations"] # how many times you want to re-check top errors overall
max_fail_passes = config["hyperparams"]["max_fail_passes"] # currently set as 2

fail_pass_count = {}
no_improvement_links = set()
# for plotting purposes, store the MSE each time it improves
improvements_list = []

early_stop = False # performance checker flag for genetic algorithm
for global_iter in range(max_global_iterations):
    # check if we converged around 27k to compare to genetic algorithm
    if early_stop:
        break
    print(f"Global iteration: {global_iter}")
    # ensures to check sorted links after improvement was found. this is because simulated volume keeps changing
    sorted_links = calculate_abs_error(link_performance_csv)
    sorted_links = [lk for lk in sorted_links if lk not in no_improvement_links] # if sorted link is in the not improved list, skip it
    if not sorted_links:
        print("No valid links remain for calibration. Exiting.")
        break  # exits the global_iter loop

    top_links = sorted_links[:20]
    print(f"Top 20 links with highest abs error: {top_links}")
    #improvement_this_cycle = False
    # iterate through top links
    for link_id in top_links:
        improvement_found = False
        if link_id in no_improvement_links:
            print(f"Skipping link {link_id} due to repeated failures.")
            continue

        # 5 attempts for each link
        for attempt in range(num_iterations):
            print(f"\nLink: {link_id}, Attempt: {attempt+1} of 5")
            sampled_od_pairs = sample_od_pairs(link_id, link_perform_odlink, current_matrix)
            if not sampled_od_pairs:
                print(f"No OD pairs found for link {link_id}. Skipping.")
                break
            # get absolute error for that link id
            simulated_vol, obs_count = get_link_data(link_id, link_performance_csv)
            if simulated_vol is None or obs_count is None:
                print(f"Skipping link {link_id} due to missing data.")
                continue

            abs_error = get_abs_error(link_id, link_performance_csv)
            prompt = model_prompt(link_id, abs_error, sampled_od_pairs, simulated_vol, obs_count)
            # pass everything to llm and output the dictionary
            llm_output_dict = generate_output(prompt, model, tokenizer, working_dir, link_id, attempt, results_path) 
            updated_pairs = llm_output_dict
            # update od matrix with new updated i,j pairs
            test_matrix = update_od_matrix(current_matrix, updated_pairs) 
            new_mse = calculate_mse(test_matrix, data_path, exe_path, results_path, link_performance_csv) # recalculate mse by running simulation
            
            # is there an improvement
            if new_mse < baseline_mse:
                print(f"Improvement found for link {link_id}: MSE improved from {baseline_mse:.4f} to {new_mse:.4f}")
                current_matrix = test_matrix.copy()
                best_matrix = current_matrix
                old_mse = baseline_mse
                baseline_mse = new_mse
                best_mse = baseline_mse
                improvements_list.append(new_mse)
                improvement_found = True
                # log values
                log_improvement_and_save(
                    results_path,
                    global_iter=global_iter,
                    link_id=link_id,
                    attempt=attempt,
                    old_mse=old_mse,
                    new_mse=new_mse,
                    updated_pairs=updated_pairs,
                    best_matrix=best_matrix
                )
                # comparison to genetic algorithm performance
                if new_mse < 27800:
                    print(f"MSE {new_mse:.4f} is below 27,800. Stopping global iterations early.")
                    early_stop = True
                #improvement_this_cycle = True
                break

        if not improvement_found:
            print(f"No improvement found for link {link_id} after {num_iterations} attempts. Moving to next link.\n")
            # increase failure count for this link
            fail_pass_count[link_id] = fail_pass_count.get(link_id, 0) + 1
            # if it has failed too many times, add to the skip set
            if fail_pass_count[link_id] >= max_fail_passes:
                no_improvement_links.add(link_id)
        else:
            print(f"Improvement found for link {link_id}. Moving on.")
            #break
    """ if not improvement_this_cycle:
        print(f"No improvements in global iteration {global_iter}. Stopping calibration.")
        break """

mse_val_path = os.path.join(results_path, "improvements.txt")
with open(mse_val_path, "a+") as f:
    for val in improvements_list:
        f.write(str(val) + "\n")

print("Recorded MSE improvements:", improvements_list)
print("LLM optimization finished.")
print(f"Final best MSE: {best_mse}")