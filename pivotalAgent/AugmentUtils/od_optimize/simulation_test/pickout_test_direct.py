import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import subprocess
import os
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2 as GA
import datetime
# from pymoo.factory import get_termination
from utils import torch2demand, faster_execute_simulation, heuristic_rule, pick_out_test


# The time consumption for execution example is about: 8 mins 

"""
This script is used to re-produce the picked out solution

directly using the saved files.

"""
curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

working_dir = "./pivotalAgent/AugmentUtils/od_optimize/simulation_test/output/"+curr_time+"/"
os.makedirs(working_dir)
save_path = working_dir + "images/"
os.makedirs(save_path)

# real observation data
real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])

current_path = os.getcwd() + "/pivotalAgent/AugmentUtils/od_optimize/simulation_test"
print(current_path)
# bash commands
commands = f"""
export SUMO_HOME=/home/local/ASURITE/longchao/Desktop/project/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
cd {current_path}
"""

# Run commands to make sure the sumo env is okay
subprocess.run(commands, shell=True)

# -----------------------------------------------------
# now we have the path to: od.gen.xml

# for now, we have fixed light duration, which is defined in networkfile no need extra control: so it is False

left_lane, right_lane = faster_execute_simulation(save_file=True, 
                                                  save_path=working_dir, 
                                                  light_control=False, 
                                                  sumoConfig_path="./pivotalAgent/AugmentUtils/od_optimize/simulation_test/light_control_3compare/sedona.sumocfg",
                                                  set_controller="libsumo",
                                                  calculate_zone6=True,
                                                  other_zones=False)

observed_sum_lane = np.array(left_lane) + np.array(right_lane)

mse = np.mean((observed_sum_lane[0:15] - real_data[0:15]) ** 2)
print("MSE:", mse)



