from DLSim import DLSim
import yaml

config_path = "/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Configs/path.yaml"

with open(config_path, 'r') as file:
    work_dic = yaml.safe_load(file)["DLSim"]["work_dic"]

# load the DLSim class
DL = DLSim()

# check the working directory
DL.check_working_directory()

# check all the required files exist
DL.check_DLSim_input_files()

# load and update settings
# DL.DLSim_settings
DL.check_working_directory(work_dic)

# perform kernel network assignment simulation
DL.perform_kernel_network_assignment_simulation()
