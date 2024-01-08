import os
import sys
import yaml
sumo_path = yaml.load(open('./pivotalAgent/Configs/path.yaml'), Loader=yaml.FullLoader)['SUMO_PATH']
# /home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Configs/path.yaml
sys.path.append(sumo_path)
# import traci  # Not used with libsumo
# import sumolib  # Not typically required for libsumo, unless for non-control utilities
import traci
import libsumo
import sumolib
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
import ast
import shapely.geometry
import re
from control_policies import control_traffic_lights, control_traffic_lights2, control_traffic_light_based_on_waiting_vehicles

def plot_count_data(left_lane, right_lane):


    # real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
    real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])
    x_data = np.arange(6, 21)

    left_lane_observe = np.array(left_lane)
    right_lane_observe = np.array(right_lane)

    # Output the result
    print("Left Lane Observations:", left_lane_observe)
    print("Right Lane Observations:", right_lane_observe)

    plt.xlabel("time/h")
    plt.ylabel("vehicle count")
    plt.title("Simulation info across time on SUMO - Sedona")
    plt.plot(x_data, real_data, label="real", color = 'r')
    plt.plot(x_data, left_lane_observe, label="left_lane")
    plt.plot(x_data, right_lane_observe, label="right_lane")
    plt.plot(x_data, np.array(left_lane_observe)+np.array(right_lane_observe), label="total pass count")
    plt.legend()
    plt.show()

def demand_16_7_7(path):
    with open(path, 'r') as file:
        data = file.read()

    # Splitting the data into blocks, each representing a 2D matrix
    blocks = data.strip().split('\n\n')

    # Processing each block
    processed_data = []
    for block in blocks:
        lines = block.strip().split('\n')
        matrix = []
        for line in lines:
            # Removing brackets and extra spaces
            cleaned_line = line.strip('[] ').replace('  ', ' ')
            # Converting the line to a list of floats
            matrix.append([float(num) for num in cleaned_line.split()])

        processed_data.append(matrix)

    OD_demand = np.transpose(processed_data, (1, 2, 0))

    # Converting the processed data into a numpy array
    return np.array(OD_demand)

def pick_out_test(path):
    with open(file=path, mode="r") as rf:
        # data = rf.read().replace("\n", " ").replace(" ", ",").replace(",,", ",")
        # data_list = ast.literal_eval(data)
        # print(len(data_list))
        # numpy_data = np.array(data_list)
        data_string = rf.read()
        data_string_cleaned = re.sub(r"[\[\]]", "", data_string).strip()
        # Convert the string to a list of floats
        data_list = [float(num) for num in data_string_cleaned.split()]
        # Convert the list to a numpy array
        data_array = np.array(data_list)

        return data_array

def combine_history(history_path):
    with open(file=history_path, mode="r") as rf:
        data_all = rf.read().replace("\n", " ").replace(" ", ",").replace(",,", ",").split("-")
        data_store = []
        for d in data_all:
            data_list = ast.literal_eval(d)
            data_store.append(data_list)

        print(len(data_list))
        numpy_data = np.array(data_list)

        return numpy_data

def experience_generate(demand_init, bias=False):

    bias_v = 0
    # demand_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
    demand_data = np.array([202, 331, 495, 683, 868, 876, 885, 906, 904, 823, 794, 685, 610, 386, 255, 155])
    # changed values
    # demand_data = np.array([59, 141, 145, 243, 348, 316, 295, 288, 270, 253, 216, 195, 180, 116, 115, 20])

    # basis to tune frOM:
    # for o in range(0, 16):
    #     if bias:
    #         bias_v = o
    #     # Generate and add new 'tazRelation' elements inside the 'intervcal'
    #     for i in range(0, 7):
    #         for j in range(0, 7):
    #             if i != j:
    #                 # if destination j is 2 (definatly pass obs poiint):
    #                 if j == 2:
    #                     if i in [0, 3, 4, 6, 5]:
    #                         demand_init[i][j][o] = demand_data[o] // 5
    #                     else:
    #                         demand_init[i][j][o] = demand_data[o] // 10 #(randomly generated value for 1)
    #                 # also if going to zone1/zone0, also possibly to pass obs point
    #                 elif j == 1 or j == 0:
    #                     demand_init[i][j][o] = demand_data[o] // 10
                    
    #                 else:
    #                     demand_init[i][j][o] = demand_data[o] // 20
    for o in range(0, 16):
        if bias:
            bias_v = o
        # Generate and add new 'tazRelation' elements inside the 'intervcal'
        for i in range(0, 7):
            for j in range(0, 7):
                if i != j:
                    
                    if i == 6 and j == 2:
                        demand_init[i][j][o] = demand_data[o] // 5 * 5
                    
                    # if i == 5 and j == 2:
                    #     demand_init[i][j][o] = demand_data[o] // 1

                    else:
                        demand_init[i][j][o] = demand_data[o] // 15


                    # if destination j is 2 (definatly pass obs poiint):
                    # if o < 5 or o >= 10:
                    #     if j == 2:
                    #         if i == 6:
                    #             demand_init[i][j][o] = demand_data[o] // 5 * 30
                    #         elif i in [0, 5]: # decrease from 0 to 1
                    #             demand_init[i][j][o] = demand_data[o] // 10
                    #         elif i in [3, 4]:
                    #             demand_init[i][j][o] = demand_data[o] // 10
                    #         else:
                    #             demand_init[i][j][o] = demand_data[o] // 10 #(randomly generated value for 1)
                    #     # also if going to zone1/zone0, also possibly to pass obs point
                    #     elif j == 0:
                    #         demand_init[i][j][o] = demand_data[o] // 40 
                        
                    #     elif j in [1, 3]:
                    #         demand_init[i][j][o] = demand_data[o] // 40
                    #     else:
                    #         demand_init[i][j][o] = demand_data[o] // 40
                    
                    # elif o >= 5 and o < 10:

                    #     if j == 2:
                    #         if i == 6:
                    #             demand_init[i][j][o] = demand_data[o] // 5 * 20
                    #         elif i in [0, 5]: # decrease from 0 to 1
                    #             demand_init[i][j][o] = demand_data[o] // 20
                    #         elif i in [3, 4]:
                    #             demand_init[i][j][o] = demand_data[o] // 20
                    #         else:
                    #             demand_init[i][j][o] = demand_data[o] // 10 #(randomly generated value for 1)
                    #     # also if going to zone1/zone0, also possibly to pass obs point
                    #     elif j == 0:
                    #         demand_init[i][j][o] = demand_data[o] // 20 
                        
                    #     elif j in [1, 3]:
                    #         demand_init[i][j][o] = demand_data[o] // 20
                    #     else:
                    #         demand_init[i][j][o] = demand_data[o] // 20
                        

            

    # even_cell = demand_data // 4 // 6 
    # # for taz 4:
    # for o in range(0, 16):
    #     if bias:
    #         bias_v = o
    #     # Generate and add new 'tazRelation' elements inside the 'intervcal'
    #     for i in range(0, 7):
    #         for j in range(0, 7):
    #             if i != j:
                    
    #                 # if i in [2, 3, 5, 6]:
    #                 #     if (i == 2 and j in [3, 5, 6]) or (i == 3 and j in [5, 6, 2]) or (i == 5 and j in [0, 1, 2, 3]) or (i == 6 and j in [0, 1, 2, 3, 5]):
    #                 if i == 6 and j == 2: # 2 is the target area
    #                     demand_init[i][j][o] = even_cell[o] * 100 + bias_v
                    
    #                 elif j == 6: # reduce direct to region 6:
    #                     demand_init[i][j][o] = 5 + bias_v
                    
    #                 elif j == 0:
    #                     demand_init[i][j][o] = even_cell[o] * 90 + bias_v
                    
    #                 else:
    #                     demand_init[i][j][o] = even_cell[o] +  bias_v

    # classic one for taz1:
    # for o in range(0, 16):
    #     if bias:
    #         bias_v = o
    #     # Generate and add new 'tazRelation' elements inside the 'intervcal'
    #     for i in range(0, 7):
    #         for j in range(0, 7):
    #             if i != j:
    #                 # if i in [2, 3, 5, 6]:
    #                 #     if (i == 2 and j in [3, 5, 6]) or (i == 3 and j in [5, 6, 2]) or (i == 5 and j in [0, 1, 2, 3]) or (i == 6 and j in [0, 1, 2, 3, 5]):
    #                 if i == 6 and j == 1:
    #                     demand_init[i][j][o] = even_cell[o] * 100 + bias_v
                    
    #                 elif j == 6: # reduce direct to region 6:
    #                     demand_init[i][j][o] = 5 + bias_v
                    
    #                 else:
    #                     demand_init[i][j][o] = 10 - bias_v


    # saved version:
    # even_cell = demand_data // 4 // 6 * 3700

    # for o in range(0, 16):
    #    if bias:
    #        bias_v = o
    #    # Generate and add new 'tazRelation' elements inside the 'intervcal'
    #    for i in range(0, 7):
    #        for j in range(0, 7):
    #            if i != j:
    #                # if i in [2, 3, 5, 6]:
    #                #     if (i == 2 and j in [3, 5, 6]) or (i == 3 and j in [5, 6, 2]) or (i == 5 and j in [0, 1, 2, 3]) or (i == 6 and j in [0, 1, 2, 3, 5]):
    #                if i == 6 and j == 1:
    #                    demand_init[i][j][o] = even_cell[o] + bias_v
                  
    #                elif j == 6: # reduce direct to region 6:
    #                    demand_init[i][j][o] = 5 + bias_v
                  
    #                else:
    #                    demand_init[i][j][o] = np.random.randint(0, 15) - bias_v


# latest version
    # for o in range(0, 16):
    #     if bias:
    #         bias_v = o
    #     # Generate and add new 'tazRelation' elements inside the 'intervcal'
    #     for i in range(0, 7):
    #         for j in range(0, 7):
    #             if i != j:
    #                 # if i in [2, 3, 5, 6]:
    #                 #     if (i == 2 and j in [3, 5, 6]) or (i == 3 and j in [5, 6, 2]) or (i == 5 and j in [0, 1, 2, 3]) or (i == 6 and j in [0, 1, 2, 3, 5]):
    #                 if i == 6:
    #                     if j == 1:
    #                         demand_init[i][j][o] = even_cell[o] * 95 + bias_v
    #                     if j == 4:
    #                         demand_init[i][j][o] = 3
    #                 elif i ==1:
    #                     # add from z1 -> z0
    #                     if j == 0:
    #                         demand_init[i][j][o] = even_cell[o] * 5
    #                     # add from z3 -> z0
    #                     if j == 3:
    #                         demand_init[i][j][o] = even_cell[o] * 2
    #                 else:
    #                     if j == 6: # reduce direct to region 6:
    #                         demand_init[i][j][o] = 5 + bias_v
    #                     else:
    #                         demand_init[i][j][o] = np.random.randint(0, 15) - bias_v

    

    return demand_init

def heuristic_rule(pop=1):

    if pop==1:

        demand_init = np.zeros((7, 7, 16))

        demand_update = experience_generate(demand_init)
        
        return demand_update
    
    else:
        solution_population = []

        for p in range(pop):

            demand_init = np.zeros((7, 7, 16))

            demand_update = experience_generate(demand_init, bias=True)
            
            solution_population.append(demand_update)
    
        return solution_population

def torch2demand(OD_demand, set_path = None):

    # Initialize a tensor with random data for the sake of example
    # In practice, this would be your actual OD demand data tensor

    # Create the root 'data' element with the namespace and schema location
    root = ET.Element("data", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/datamode_file.xsd"
    })

    # Function to create a 'tazRelation' element
    def create_taz_relation(from_zone, to_zone, count):
        taz_relation = ET.Element("tazRelation")
        taz_relation.set("from", from_zone)
        taz_relation.set("to", to_zone)
        taz_relation.set("count", str(count))
        return taz_relation

    # Create the 'interval' elements and populate them with 'tazRelation' data from the tensor
    for o in range(16):  # Assuming 'o' is the index for hours
        interval = ET.SubElement(root, "interval", begin=f"{o}:0:0", end=f"{o+1}:0:0")
        for i in range(7):  # Origin zones
            for j in range(7):  # Destination zones
                if i != j:  # Assuming no self-transitions
                    count = OD_demand[i, j, o]
                    from_zone = f"taz_{i}"
                    to_zone = f"taz_{j}"
                    taz_relation = create_taz_relation(from_zone, to_zone, count)
                    interval.append(taz_relation)

    # Convert the modified tree to a string and pretty-print
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    save_path = './demand_info/od.gen.xml'
    # Write the pretty-printed XML to the file
    if set_path != None:
        hard_path = set_path 
    else: 
        hard_path = "/home/local/ASURITE/longchao/Desktop/project/sedona/pure_simulation/optimize_process/demand_info/od.gen.xml"
    with open(hard_path, 'w') as file:
        file.write(reparsed.toprettyxml(indent="  "))
    
    return save_path


# def execute_simulation():

    viusalize = True
    controller = "libsumo"
    interval_duration = 3600

    # Define the path to your configuration file
    sumoConfig = "./demand_info/sedona.sumocfg"

    if controller == "libsumo":
        control = libsumo
        control.start(["-c", sumoConfig])
    elif controller == "traci":
        control = traci
        sumoBinary = sumolib.checkBinary('sumo')
        control.start([sumoBinary, "-c", sumoConfig])

    # Your existing setup code...

    # Initialize lists to hold the observation counts for each lane
    left_lane_observe = []
    right_lane_observe = []

    # Initialize counters for the current interval
    current_left_lane_count = 0
    current_right_lane_count = 0

    # Set the interval duration

    # Store vehicle delays
    vehicle_delays = {}
    vehicle_travel_times = {}
    vehicle_start_times = {}

    total = 57600
    pbar = tqdm(total)
    # Run the simulation
    step = 0
    start_time = datetime.datetime.now()


    # use detector in python:
    detectors = {
        "detector1": {"lane": "1089324756#1_1", "position": 7.70},
        "detector2": {"lane": "1089324756#1_0", "position": 8.19}
    }

    # Initialize a dictionary to hold vehicle counts
    counted_vehicles = {"detector1": set(), "detector2": set()}

    # Run the simulation
    while step < total:
        control.simulationStep()  # Advance the simulation

        # Detecting vehicles...
        for detector_id, detector_info in detectors.items():
            vehicles_on_lane = control.lane.getLastStepVehicleIDs(detector_info["lane"])
            for vehicle in vehicles_on_lane:
                pos = control.vehicle.getLanePosition(vehicle)
                if pos >= detector_info["position"] and pos < detector_info["position"] + 10:

                    # avoid when traffic is jammed, multiple counts over one vechile 
                    if vehicle not in counted_vehicles[detector_id]:
                    
                        if detector_id == "detector1":
                            current_left_lane_count += 1
                        elif detector_id == "detector2":
                            current_right_lane_count += 1
                        
                        # Mark the vehicle as counted
                        counted_vehicles[detector_id].add(vehicle)

        # Check if the current interval has ended
        if step % interval_duration == 0 and step != 0:
            # Store the counts in the respective lists and reset counters
            left_lane_observe.append(current_left_lane_count)
            right_lane_observe.append(current_right_lane_count)
            current_left_lane_count = 0
            current_right_lane_count = 0

        # Your existing vehicle delay and travel time calculation code...

        step += 1
        pbar.update(1)

    # Include counts for the last interval if it's not exactly 3600 seconds
    if step % interval_duration != 0:
        left_lane_observe.append(current_left_lane_count)# Initialize a tensor with random data for the sake of example
# In practice, this would be your actual OD demand data tensor
# OD_demand = torch.randint(0, 100, (7, 7, 16))

# Create the root 'data' element with the namespace and schema location
# root = ET.Element("data", {
#     "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
#     "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/datamode_file.xsd"
# })

# Function to create a 'tazRelation' element
def create_taz_relation(from_zone, to_zone, count):
    taz_relation = ET.Element("tazRelation")
    taz_relation.set("from", from_zone)
    taz_relation.set("to", to_zone)
    taz_relation.set("count", str(count))
    return taz_relation

# # Create the 'interval' elements and populate them with 'tazRelation' data from the tensor
# for o in range(16):  # Assuming 'o' is the index for hours
#     interval = ET.SubElement(root, "interval", begin=f"{o}:0:0", end=f"{o+1}:0:0")
#     for i in range(7):  # Origin zones
#         for j in range(7):  # Destination zones
#             if i != j:  # Assuming no self-transitions
#                 count = OD_demand[i, j, o].item()  # Retrieve the count from the tensor
#                 from_zone = f"taz_{i}"
#                 to_zone = f"taz_{j}"
#                 taz_relation = create_taz_relation(from_zone, to_zone, count)
#                 interval.append(taz_relation)

# # Convert the modified tree to a string and pretty-print
# rough_string = ET.tostring(root, 'utf-8')
# reparsed = minidom.parseString(rough_string)

# # Write the pretty-printed XML to the file
# hard_path = "/home/local/ASURITE/longchao/Desktop/project/sedona/pure_simulation/optimize_process/demand_info/od.gen.xml"
# relative_path = "./demand_info/od.gen.xml"
# with open(hard_path, 'w') as file:
#     file.write(reparsed.toprettyxml(indent="  "))

# Function to parse the TAZ shape string into a list of tuples
def parse_taz_shape(shape_str):

    # Split the string by spaces to separate each coordinate pair
    coords = shape_str.split()
    # Convert each pair of coordinates to a tuple of floats
    return [tuple(map(float, coord.split(','))) for coord in coords]


def faster_execute_simulation(save_file=False, viusalize = True, save_path="./", para_set=0, light_control=False, sumoConfig_path=None, set_controller = "libsumo",  calculate_zone6=False, detailed_record=True, other_zones = False, extra_zone=False):
    """
    Args:
        save_file (bool, optional): [description]. Defaults to False.
        viusalize (bool, optional): [description]. Defaults to True.
        save_path (str, optional): [description]. Defaults to "./".
        para_set (int, optional): [description]. Defaults to 0.
        light_control (bool, optional): [description]. Defaults to False.
        sumoConfig_path ([type], optional): [description]. Defaults to None.
        set_controller (str, optional): [description]. Defaults to "libsumo".
        calculate_zone6 (bool, optional): [description]. Defaults to False.
        detailed_record (bool, optional): [description]. Defaults to True.
        other_zones (bool, optional): [description]. Defaults to False.
        extra_zone (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    viusalize = viusalize
    save = save_file
    controller = set_controller
    interval_duration = 3600
    calculate_zone6_level = calculate_zone6
    detailed_record = detailed_record
    extra_zone = extra_zone

    other_zones = other_zones

    if sumoConfig_path != None:
        sumoConfig = sumoConfig_path
    else:
        # Define the path to your configuration file
        sumoConfig = "./demand_info/sedona.sumocfg"

    if controller == "libsumo":
        control = libsumo
        control.start(["-c", sumoConfig])
    elif controller == "traci":
        control = traci
        sumoBinary = sumolib.checkBinary('sumo-gui')
        # sumoBinary = sumolib.checkBinary('sumo')
        control.start([sumoBinary, "-c", sumoConfig])

    # Your existing setup code...

    # Initialize lists to hold the observation counts for each lane
    left_lane_observe = []
    right_lane_observe = []

    left_conflict_obsr = []
    right_conflict_obsr = []

    # delay_hourly:
    hourly_delay = []
    hourly_travel = []


    # Initialize counters for the current interval
    current_left_lane_count = 0
    current_right_lane_count = 0

    # conflict
    curr_left_conflict = 0
    curr_right_conflict = 0

    # Set the interval duration

    # Store vehicle delays
    vehicle_delays = {}
    vehicle_travel_times = {}
    vehicle_start_times = {}

    # entry the zone: for zone level travel time
    vehicle_entry_times = {}

    # taz6
    taz6_vehicle_travel_times = {}
    taz6_vehicle_delays = {}
    vehicles_in_taz6 = set()
    hourly_zone6_delay = []
    hourly_zone6_travel_time = []



    # taz0:
    vehicles_in_taz0 = set()
    taz0_vehicle_travel_times = {}
    taz0_vehicle_delays = {}
    hourly_zone0_delay = []
    hourly_zone0_travel_time = []

    # taz1:
    vehicles_in_taz1 = set()
    taz1_vehicle_travel_times = {}
    taz1_vehicle_delays = {}
    hourly_zone1_delay = []
    hourly_zone1_travel_time = []

    # taz4:
    vehicles_in_taz4 = set()
    taz4_vehicle_travel_times = {}
    taz4_vehicle_delays = {}
    hourly_zone4_delay = []
    hourly_zone4_travel_time = []

    # taz extra:
    vehicles_in_taz_e = set()
    taz_e_vehicle_travel_times = {}
    taz_e_vehicle_delays = {}
    hourly_zone_e_delay = []
    hourly_zone_e_travel_time = []


    total = 57600
    pbar = tqdm(total)
    # Run the simulation
    step = 0
    start_time = datetime.datetime.now()

    # after prune:
    #  <inductionLoop id="e1_1" name="count1" lane="1089324756#1_1" pos="14.65" period="300.00" file="detector1.xml"/>
    # <inductionLoop id="e1_2" name="count2" lane="1089324756#1_0" pos="15.55" period="300.00" file="detector2.xml"/>
    # <inductionLoop id="e2_1" name="conflict1" lane="772044499_1" pos="2.37" period="300.00" file="conflict1.xml"/>
    # <inductionLoop id="e2_2" name="conflict2" lane="772044499_0" pos="1.68" period="300.00" file="conflict2.xml"/>
    
    detectors = {
        "detector1": {"lane": "1089324756#1_1", "position": 14.65},
        "detector2": {"lane": "1089324756#1_0", "position": 15.55},
        "conflict1": {"lane": "772044499_1", "position": 2.37},
        "conflict2": {"lane": "772044499_0", "position": 1.68},
    }
    # before prune:
    # use detector in python:
    # detectors = {
    #     "detector1": {"lane": "1089324756#1_1", "position": 7.70},
    #     "detector2": {"lane": "1089324756#1_0", "position": 8.19},
    #     "conflict1": {"lane": "772044499_1", "position": 4.16},
    #     "conflict2": {"lane": "772044499_0", "position": 2.82},
    # }

    # Initialize a dictionary to hold vehicle counts

    counted_vehicles = {"detector1": set(), "detector2": set(), "conflict1": set(), "conflict2": set()}


    # Run the simulation
    while step < total:
        control.simulationStep()  # Advance the simulation

        if calculate_zone6_level:
            # thuis is the extra area:
            # polygon6 = [(1155.68, 1375.54), (1106.26, 1220.89), (1500.84, 782.47), (1588.3, 635.7), (1757.76, 984.89), (1560.06, 1431.65), (1156.95, 1373.88), (1155.68, 1375.54)]
            
            # true 6:
            polygon6 = [(1331.45, 1205.31), (1320.95, 1138.78), (1333.79, 1012.74), (1294.11, 861.02), (1341.96, 645.11), (1429.49, 603.09), (1477.34, 674.29), (1499.51, 753.65), (1561.37, 783.99), (1555.53, 851.68), (1462.17, 973.06), (1511.18, 1169.13), (1369.97, 1226.32), (1331.45, 1202.97), (1331.45, 1205.31)]
            taz_6_polygon = shapely.geometry.Polygon(polygon6)
            
            for veh_id in control.vehicle.getIDList():
                x, y = control.vehicle.getPosition(veh_id)
                point = shapely.geometry.Point(x, y)
                if taz_6_polygon.contains(point):
                    if veh_id not in vehicles_in_taz6:
                        vehicles_in_taz6.add(veh_id)
                        vehicle_entry_times[veh_id] = control.simulation.getTime()
                else:
                    if veh_id in vehicles_in_taz6:
                        vehicles_in_taz6.remove(veh_id)
                        travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                        taz6_vehicle_travel_times[veh_id] = travel_time
                        del vehicle_entry_times[veh_id]

            active_vehicle_ids = set(control.vehicle.getIDList())
            discard_list = []
            for veh_id in vehicles_in_taz6:
                if veh_id in active_vehicle_ids:
                    # print("veh_id:")
                    # print(veh_id)
                    delay = control.vehicle.getAccumulatedWaitingTime(veh_id)
                    taz6_vehicle_delays[veh_id] = delay

                    # get the average travel time for vechicle in this zone:
                    travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                    taz6_vehicle_travel_times[veh_id] = travel_time
                else:
                    discard_list.append(veh_id)
            for id in discard_list:
                vehicles_in_taz6.discard(id)
        
        if other_zones:
            # ===========0=============
            polygon0 = [(11.5, 1187.18), (34.06, 860.13), (940.03, 1311.24), (960.7, 1600.7), (9.62, 1183.43), (11.5, 1187.18)]            
            
            
            taz_0_polygon = shapely.geometry.Polygon(polygon0)
            
            for veh_id in control.vehicle.getIDList():
                x, y = control.vehicle.getPosition(veh_id)
                point = shapely.geometry.Point(x, y)
                if taz_0_polygon.contains(point):
                    if veh_id not in vehicles_in_taz0:
                        vehicles_in_taz0.add(veh_id)
                        vehicle_entry_times[veh_id] = control.simulation.getTime()
                else:
                    if veh_id in vehicles_in_taz0:
                        vehicles_in_taz0.remove(veh_id)
                        travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                        taz0_vehicle_travel_times[veh_id] = travel_time
                        del vehicle_entry_times[veh_id]

            active_vehicle_ids = set(control.vehicle.getIDList())
            discard_list = []
            for veh_id in vehicles_in_taz0:
                if veh_id in active_vehicle_ids:
                    # print("veh_id:")
                    # print(veh_id)
                    delay = control.vehicle.getAccumulatedWaitingTime(veh_id)
                    taz0_vehicle_delays[veh_id] = delay

                    # get the average travel time for vechicle in this zone:
                    travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                    taz0_vehicle_travel_times[veh_id] = travel_time
                else:
                    discard_list.append(veh_id)
            for id in discard_list:
                vehicles_in_taz0.discard(id)


            # ============taz1============

            polygon1 = [(1233.93, 1566.03), (1321.62, 1289.94), (1354.95, 1276.89), (1462.93, 1293.56), (1540.47, 1410.23), (1575.25, 1500.81), (1565.1, 1580.53), (1233.93, 1576.9), (1233.93, 1566.03)]
            
            
            taz_1_polygon = shapely.geometry.Polygon(polygon1)
            
            for veh_id in control.vehicle.getIDList():
                x, y = control.vehicle.getPosition(veh_id)
                point = shapely.geometry.Point(x, y)
                if taz_1_polygon.contains(point):
                    if veh_id not in vehicles_in_taz1:
                        vehicles_in_taz1.add(veh_id)
                        vehicle_entry_times[veh_id] = control.simulation.getTime()
                else:
                    if veh_id in vehicles_in_taz1:
                        vehicles_in_taz1.remove(veh_id)
                        travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                        taz1_vehicle_travel_times[veh_id] = travel_time
                        del vehicle_entry_times[veh_id]

            active_vehicle_ids = set(control.vehicle.getIDList())
            discard_list = []
            for veh_id in vehicles_in_taz1:
                if veh_id in active_vehicle_ids:
                    # print("veh_id:")
                    # print(veh_id)
                    delay = control.vehicle.getAccumulatedWaitingTime(veh_id)
                    taz1_vehicle_delays[veh_id] = delay

                    # get the average travel time for vechicle in this zone:
                    travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                    taz1_vehicle_travel_times[veh_id] = travel_time
                else:
                    discard_list.append(veh_id)
            for id in discard_list:
                vehicles_in_taz1.discard(id)

            # ===========taz4=============

            polygon4 = [(1533.44, 723.94), (1290.55, 262.16), (1357.21, 88.48), (1519.82, 111.0), (1684.94, 766.46), (1532.33, 721.43), (1533.44, 723.94)]
            
            
            taz_4_polygon = shapely.geometry.Polygon(polygon4)
            
            for veh_id in control.vehicle.getIDList():
                x, y = control.vehicle.getPosition(veh_id)
                point = shapely.geometry.Point(x, y)
                if taz_4_polygon.contains(point):
                    if veh_id not in vehicles_in_taz4:
                        vehicles_in_taz4.add(veh_id)
                        vehicle_entry_times[veh_id] = control.simulation.getTime()
                else:
                    if veh_id in vehicles_in_taz4:
                        vehicles_in_taz4.remove(veh_id)
                        travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                        taz4_vehicle_travel_times[veh_id] = travel_time
                        del vehicle_entry_times[veh_id]

            active_vehicle_ids = set(control.vehicle.getIDList())
            discard_list = []
            for veh_id in vehicles_in_taz4:
                if veh_id in active_vehicle_ids:
                    # print("veh_id:")
                    # print(veh_id)
                    delay = control.vehicle.getAccumulatedWaitingTime(veh_id)
                    taz4_vehicle_delays[veh_id] = delay

                    # get the average travel time for vechicle in this zone:
                    travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                    taz4_vehicle_travel_times[veh_id] = travel_time
                else:
                    discard_list.append(veh_id)
            for id in discard_list:
                vehicles_in_taz4.discard(id)

            # ===========taz extra=============
        if extra_zone:
            polygon_e = [(1155.68, 1375.54), (1106.26, 1220.89), (1500.84, 782.47), (1588.3, 635.7), (1757.76, 984.89), (1560.06, 1431.65), (1156.95, 1373.88), (1155.68, 1375.54)]
            
            
            taz_e_polygon = shapely.geometry.Polygon(polygon_e)
            
            for veh_id in control.vehicle.getIDList():
                x, y = control.vehicle.getPosition(veh_id)
                point = shapely.geometry.Point(x, y)
                if taz_e_polygon.contains(point):
                    if veh_id not in vehicles_in_taz_e:
                        vehicles_in_taz_e.add(veh_id)
                        vehicle_entry_times[veh_id] = control.simulation.getTime()
                else:
                    if veh_id in vehicles_in_taz_e:
                        vehicles_in_taz_e.remove(veh_id)
                        travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                        taz_e_vehicle_travel_times[veh_id] = travel_time
                        del vehicle_entry_times[veh_id]

            active_vehicle_ids = set(control.vehicle.getIDList())
            discard_list = []
            for veh_id in vehicles_in_taz_e:
                if veh_id in active_vehicle_ids:
                    # print("veh_id:")
                    # print(veh_id)
                    delay = control.vehicle.getAccumulatedWaitingTime(veh_id)
                    taz_e_vehicle_delays[veh_id] = delay

                    # get the average travel time for vechicle in this zone:
                    travel_time = control.simulation.getTime() - vehicle_entry_times[veh_id]
                    taz_e_vehicle_travel_times[veh_id] = travel_time
                else:
                    discard_list.append(veh_id)
            for id in discard_list:
                vehicles_in_taz_e.discard(id)



        if light_control:
            # control_traffic_lights2(control=control)
            control_traffic_light_based_on_waiting_vehicles(control=control, tlID="1295124487", laneID1="141298282#0_0", laneID2="141298282#0_1")
            control_traffic_light_based_on_waiting_vehicles(control=control, tlID="1546745521", laneID1="141298279_0", laneID2="141298279_1")
            # down below:
            control_traffic_light_based_on_waiting_vehicles(control=control, tlID="1093668191", laneID1="141298276#4_0", laneID2="141298276#4_1")
            # control_traffic_lights(control=control)

        if detailed_record:
            # Get list of vehicle IDs
            vehicle_ids = control.vehicle.getIDList()

            for v_id in vehicle_ids:
                # Get the vehicle's current delay
                delay = control.vehicle.getAccumulatedWaitingTime(v_id)
                # update the accumulated delay
                vehicle_delays[v_id] = delay
                
                # Mark the departure time for each vehicle
                if v_id not in vehicle_start_times:
                    vehicle_start_times[v_id] = control.simulation.getTime()
            
            # Process vehicles that have left the simulation
            finished_vehicles = set(vehicle_start_times) - set(vehicle_ids)
            for v_id in finished_vehicles:
                # Vehicle has finished its route and is no longer in the simulation
                arrival_time = control.simulation.getTime()
                travel_time = arrival_time - vehicle_start_times[v_id]
                vehicle_travel_times[v_id] = travel_time
                del vehicle_start_times[v_id]

        # Detecting vehicles...
        for detector_id, detector_info in detectors.items():
            vehicles_on_lane = control.lane.getLastStepVehicleIDs(detector_info["lane"])
            for vehicle in vehicles_on_lane:
                pos = control.vehicle.getLanePosition(vehicle)
                if pos >= detector_info["position"] and pos < detector_info["position"] + 10:

                    # avoid when traffic is jammed, multiple counts over one vechile 
                    if vehicle not in counted_vehicles[detector_id]:
                    
                        if detector_id == "detector1":
                            current_left_lane_count += 1
                        elif detector_id == "detector2":
                            current_right_lane_count += 1
                        elif detector_id == "conflict1":
                            curr_left_conflict += 1
                        elif detector_id == "conflict2":
                            curr_right_conflict += 1
                        
                        # Mark the vehicle as counted
                        counted_vehicles[detector_id].add(vehicle)
                        
        # Check if the current interval has ended: 3600 for interval
        if step % interval_duration == 0 and step != 0:
            # Store the counts in the respective lists and reset counters
            left_lane_observe.append(current_left_lane_count)
            right_lane_observe.append(current_right_lane_count)

            left_conflict_obsr.append(curr_left_conflict)
            right_conflict_obsr.append(curr_right_conflict)
            
            if detailed_record:
                average_delay_temp = sum(vehicle_delays.values()) / len(vehicle_delays)
                hourly_delay.append(average_delay_temp)
                # reset the delays into 0 and start accumulating again:
                # vehicle_delays = {key: 0 for key in vehicle_delays}

                average_travel_time_temp = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
                hourly_travel.append(average_travel_time_temp)
                # vehicle_travel_times = {key : 0 for key in vehicle_travel_times}

                active_vehicle_ids = set(control.vehicle.getIDList())

                for veh_id in list(vehicle_delays.keys()):
                    if veh_id not in active_vehicle_ids:
                        del vehicle_delays[veh_id]

                for veh_id in list(vehicle_travel_times.keys()):
                    if veh_id not in active_vehicle_ids:
                        del vehicle_travel_times[veh_id]

            if calculate_zone6_level:
                # hourly_zone0 delay:
                average_delay_taz6_temp = sum(taz6_vehicle_delays.values()) / len(taz6_vehicle_delays) if taz6_vehicle_delays else 0
                hourly_zone6_delay.append(average_delay_taz6_temp)
                taz6_vehicle_delays = {key: 0 for key in taz6_vehicle_delays}
                exited_vehicles_1 = set(taz6_vehicle_delays.keys()) - vehicles_in_taz6
                for veh_id in exited_vehicles_1:
                    del taz6_vehicle_delays[veh_id]
                
                average_travel_time_taz6_temp = sum(taz6_vehicle_travel_times.values()) / len(taz6_vehicle_travel_times) if taz6_vehicle_travel_times else 0
                hourly_zone6_travel_time.append(average_travel_time_taz6_temp)
                exited_vehicles = set(taz6_vehicle_travel_times.keys()) - vehicles_in_taz6
                for veh_id in exited_vehicles:
                    del taz6_vehicle_travel_times[veh_id]
            
            if other_zones:
                # taz0:
                average_delay_taz0_temp = sum(taz0_vehicle_delays.values()) / len(taz0_vehicle_delays) if taz0_vehicle_delays else 0
                hourly_zone0_delay.append(average_delay_taz0_temp)
                taz0_vehicle_delays = {key: 0 for key in taz0_vehicle_delays}
                exited_vehicles_1 = set(taz0_vehicle_delays.keys()) - vehicles_in_taz0
                for veh_id in exited_vehicles_1:
                    del taz0_vehicle_delays[veh_id]
                
                average_travel_time_taz0_temp = sum(taz0_vehicle_travel_times.values()) / len(taz0_vehicle_travel_times) if taz0_vehicle_travel_times else 0
                hourly_zone0_travel_time.append(average_travel_time_taz0_temp)
                exited_vehicles = set(taz0_vehicle_travel_times.keys()) - vehicles_in_taz0
                for veh_id in exited_vehicles:
                    del taz0_vehicle_travel_times[veh_id]

                #taz1:
                average_delay_taz1_temp = sum(taz1_vehicle_delays.values()) / len(taz1_vehicle_delays) if taz1_vehicle_delays else 0
                hourly_zone1_delay.append(average_delay_taz1_temp)
                taz1_vehicle_delays = {key: 0 for key in taz1_vehicle_delays}
                exited_vehicles_1 = set(taz1_vehicle_delays.keys()) - vehicles_in_taz1
                for veh_id in exited_vehicles_1:
                    del taz1_vehicle_delays[veh_id]
                
                average_travel_time_taz1_temp = sum(taz1_vehicle_travel_times.values()) / len(taz1_vehicle_travel_times) if taz1_vehicle_travel_times else 0
                hourly_zone1_travel_time.append(average_travel_time_taz1_temp)
                exited_vehicles = set(taz1_vehicle_travel_times.keys()) - vehicles_in_taz1
                for veh_id in exited_vehicles:
                    del taz1_vehicle_travel_times[veh_id]      

                #taz4:
                average_delay_taz4_temp = sum(taz4_vehicle_delays.values()) / len(taz4_vehicle_delays) if taz4_vehicle_delays else 0
                hourly_zone4_delay.append(average_delay_taz4_temp)
                taz4_vehicle_delays = {key: 0 for key in taz4_vehicle_delays}
                exited_vehicles_1 = set(taz4_vehicle_delays.keys()) - vehicles_in_taz4
                for veh_id in exited_vehicles_1:
                    del taz4_vehicle_delays[veh_id]
                
                average_travel_time_taz4_temp = sum(taz4_vehicle_travel_times.values()) / len(taz4_vehicle_travel_times) if taz4_vehicle_travel_times else 0
                hourly_zone4_travel_time.append(average_travel_time_taz4_temp)
                exited_vehicles = set(taz4_vehicle_travel_times.keys()) - vehicles_in_taz4
                for veh_id in exited_vehicles:
                    del taz4_vehicle_travel_times[veh_id]        
            if extra_zone:
                #taz e:
                average_delay_taz_e_temp = sum(taz_e_vehicle_delays.values()) / len(taz_e_vehicle_delays) if taz_e_vehicle_delays else 0
                hourly_zone_e_delay.append(average_delay_taz_e_temp)
                taz_e_vehicle_delays = {key: 0 for key in taz_e_vehicle_delays}
                exited_vehicles_1 = set(taz_e_vehicle_delays.keys()) - vehicles_in_taz_e
                for veh_id in exited_vehicles_1:
                    del taz_e_vehicle_delays[veh_id]
                
                average_travel_time_taz_e_temp = sum(taz_e_vehicle_travel_times.values()) / len(taz_e_vehicle_travel_times) if taz_e_vehicle_travel_times else 0
                hourly_zone_e_travel_time.append(average_travel_time_taz_e_temp)
                exited_vehicles = set(taz_e_vehicle_travel_times.keys()) - vehicles_in_taz_e
                for veh_id in exited_vehicles:
                    del taz_e_vehicle_travel_times[veh_id]      
                

            current_left_lane_count = 0
            current_right_lane_count = 0

            curr_left_conflict = 0
            curr_right_conflict = 0


        # Your existing vehicle delay and travel time calculation code...

        step += 1
        pbar.update(1)

    # Include counts for the last interval if it's not exactly 3600 seconds
    if step % interval_duration != 0:
        left_lane_observe.append(current_left_lane_count)
        right_lane_observe.append(current_right_lane_count)

        left_conflict_obsr.append(curr_left_conflict)
        right_conflict_obsr.append(curr_right_conflict)

    # Your existing code to calculate average delay and travel time...

    end_time = datetime.datetime.now()

    # Calculate the average delay
    if detailed_record:
        if vehicle_delays:
            # average_delay = sum(vehicle_delays.values()) / len(vehicle_delays)
            average_delay = np.mean(hourly_delay)
        else: 
            average_delay = 0

        # Calculate the average travel time
        if vehicle_travel_times:
            # average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
            average_travel_time = np.mean(hourly_travel)
        else:
            average_travel_time = 0

    # Output the result
    print("Left Lane Observations:", left_lane_observe)
    print("Right Lane Observations:", right_lane_observe)

    print("Left Lane conflict:", left_conflict_obsr)
    print("Right Lane conflict:", right_conflict_obsr)

    total_direct = np.array(left_lane_observe) + np.array(right_lane_observe)
    total_conflict = np.array(left_conflict_obsr) + np.array(right_conflict_obsr)

    up_v = total_direct
    down_v = 1420 * np.exp(-0.00085 * total_conflict)

    predicted_approach_capacity = (up_v/down_v) * 0.001

    print("predicted_approach_capacity", predicted_approach_capacity)

    if detailed_record:
        print(f"Average delay time: {average_delay} seconds")
        print(f"Average travel time: {average_travel_time} seconds")

        print(f"Average delay time (hourly): {hourly_delay} seconds")
        print(f"Average travel time (hourly): {hourly_travel} seconds")
    if calculate_zone6_level:
        print(f"Zone6: Average delay time (hourly): {hourly_zone6_delay} seconds")
        print(f"Zone6: Average travel time (hourly): {hourly_zone6_travel_time} seconds")

    if other_zones:
        print(f"Zone0: Average delay time (hourly): {hourly_zone0_delay} seconds")
        print(f"Zone0: Average travel time (hourly): {hourly_zone0_travel_time} seconds")

        print(f"Zone1: Average delay time (hourly): {hourly_zone1_delay} seconds")
        print(f"Zone1: Average travel time (hourly): {hourly_zone1_travel_time} seconds")

        print(f"Zone4: Average delay time (hourly): {hourly_zone4_delay} seconds")
        print(f"Zone4: Average travel time (hourly): {hourly_zone4_travel_time} seconds")

    if extra_zone:
        print(f"Zone extra: Average delay time (hourly): {hourly_zone_e_delay} seconds")
        print(f"Zone extra: Average travel time (hourly): {hourly_zone_e_travel_time} seconds")

    print(f"total time cost:{end_time - start_time}")
    

    # real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
    real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])
    x_data = np.arange(6, 21)

    plt.xlabel("time/h")
    plt.ylabel("vehicle count")
    plt.title("Simulation info across time on SUMO - Sedona")
    plt.plot(x_data, real_data, label="real", color = 'r')
    # plt.plot(x_data, left_lane_observe, label="left_lane")
    # plt.plot(x_data, right_lane_observe, label="right_lane")
    plt.plot(x_data, np.array(left_lane_observe)+np.array(right_lane_observe), label="total pass count")
    plt.legend()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(save_path + "images/"+ str(para_set)+"_"+curr_time, dpi=700)
        if not viusalize:
            plt.clf()
    if viusalize:
        plt.show()
        plt.clf()
    
    return left_lane_observe, right_lane_observe


def execute_simulation(save_file=False, viusalize = True, save_path=".", para_set=0, light_control=False):

    viusalize = viusalize
    save = save_file
    controller = "libsumo"
    interval_duration = 3600

    # Define the path to your configuration file
    sumoConfig = "./demand_info/sedona.sumocfg"

    if controller == "libsumo":
        control = libsumo
        control.start(["-c", sumoConfig])
    elif controller == "traci":
        control = traci
        sumoBinary = sumolib.checkBinary('sumo')
        control.start([sumoBinary, "-c", sumoConfig])

    # Your existing setup code...

    # Initialize lists to hold the observation counts for each lane
    left_lane_observe = []
    right_lane_observe = []

    # Initialize counters for the current interval
    current_left_lane_count = 0
    current_right_lane_count = 0

    # Set the interval duration

    # Store vehicle delays
    vehicle_delays = {}
    vehicle_travel_times = {}
    vehicle_start_times = {}

    total = 57600
    pbar = tqdm(total)
    # Run the simulation
    step = 0
    start_time = datetime.datetime.now()


    # use detector in python:
    detectors = {
        "detector1": {"lane": "1089324756#1_1", "position": 7.70},
        "detector2": {"lane": "1089324756#1_0", "position": 8.19}
    }

    # Initialize a dictionary to hold vehicle counts
    vehicle_counts = {"detector1": 0, "detector2": 0}

    counted_vehicles = {"detector1": set(), "detector2": set()}

    # Run the simulation
    while step < total:
        control.simulationStep()  # Advance the simulation
        if light_control:
            control_traffic_lights(control=control)

        # Get list of vehicle IDs
        vehicle_ids = control.vehicle.getIDList()

        for v_id in vehicle_ids:
            # Get the vehicle's current delay
            delay = control.vehicle.getAccumulatedWaitingTime(v_id)
            
            # Update the total delay
            vehicle_delays[v_id] = delay

            # Mark the departure time for each vehicle
            if v_id not in vehicle_start_times:
                vehicle_start_times[v_id] = control.simulation.getTime()  # libsumo returns time in seconds
            # Check for vehicles that have arrived
        
        for v_id in list(vehicle_start_times):
            if v_id not in control.vehicle.getIDList():
                # Vehicle has finished its route and is no longer in the simulation
                arrival_time = control.simulation.getTime()
                travel_time = arrival_time - vehicle_start_times[v_id]
                vehicle_travel_times[v_id] = travel_time
                # Remove the vehicle from the start times to avoid double counting
                del vehicle_start_times[v_id]
            else:
                if not control.vehicle.isRouteValid(v_id) or control.vehicle.getRoadID(v_id) == '':
                    arrival_time = control.simulation.getTime()
                    travel_time = arrival_time - vehicle_start_times[v_id]
                    vehicle_travel_times[v_id] = travel_time
                    # Remove the vehicle from the start times to avoid double counting
                    del vehicle_start_times[v_id]

        # Detecting vehicles...
        for detector_id, detector_info in detectors.items():
            vehicles_on_lane = control.lane.getLastStepVehicleIDs(detector_info["lane"])
            for vehicle in vehicles_on_lane:
                pos = control.vehicle.getLanePosition(vehicle)
                if pos >= detector_info["position"] and pos < detector_info["position"] + 10:

                    # avoid when traffic is jammed, multiple counts over one vechile 
                    if vehicle not in counted_vehicles[detector_id]:
                    
                        if detector_id == "detector1":
                            current_left_lane_count += 1
                        elif detector_id == "detector2":
                            current_right_lane_count += 1
                        
                        # Mark the vehicle as counted
                        counted_vehicles[detector_id].add(vehicle)

        # Check if the current interval has ended
        if step % interval_duration == 0 and step != 0:
            # Store the counts in the respective lists and reset counters
            left_lane_observe.append(current_left_lane_count)
            right_lane_observe.append(current_right_lane_count)
            current_left_lane_count = 0
            current_right_lane_count = 0

        # Your existing vehicle delay and travel time calculation code...

        step += 1
        pbar.update(1)

    # Include counts for the last interval if it's not exactly 3600 seconds
    if step % interval_duration != 0:
        left_lane_observe.append(current_left_lane_count)
        right_lane_observe.append(current_right_lane_count)

    # Your existing code to calculate average delay and travel time...

    end_time = datetime.datetime.now()

    # Calculate the average delay
    if vehicle_delays:
        average_delay = sum(vehicle_delays.values()) / len(vehicle_delays)
    else: 
        average_delay = 0

    # Calculate the average travel time
    if vehicle_travel_times:
        average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
    else:
        average_travel_time = 0

    # Output the result
    print("Left Lane Observations:", left_lane_observe)
    print("Right Lane Observations:", right_lane_observe)
    print(f"Average delay time: {average_delay} seconds")
    print(f"Average travel time: {average_travel_time} seconds")
    print(f"total time cost:{end_time - start_time}")
    

    # real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
    real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])
    x_data = np.arange(6, 21)

    plt.xlabel("time/h")
    plt.ylabel("vehicle count")
    plt.title("Simulation info across time on SUMO - Sedona")
    plt.plot(x_data, real_data, label="real", color = 'r')
    plt.plot(x_data, left_lane_observe, label="left_lane")
    plt.plot(x_data, right_lane_observe, label="right_lane")
    plt.plot(x_data, np.array(left_lane_observe)+np.array(right_lane_observe), label="total pass count")
    plt.legend()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(save_path + "images/"+ str(para_set)+"_"+curr_time, dpi=700)
        plt.clf()
    if viusalize:
        plt.show()
        plt.clf()
        
    return left_lane_observe, right_lane_observe


# def simulate_for_delay(save_file=False, viusalize = True, save_path=".", para_set=0):

    viusalize = viusalize
    save = save_file
    controller = "libsumo"
    interval_duration = 3600

    # Define the path to your configuration file
    sumoConfig = "./demand_info/sedona.sumocfg"

    if controller == "libsumo":
        control = libsumo
        control.start(["-c", sumoConfig])
    elif controller == "traci":
        control = traci
        sumoBinary = sumolib.checkBinary('sumo')
        control.start([sumoBinary, "-c", sumoConfig])

    # Your existing setup code...

    # Initialize lists to hold the observation counts for each lane
    left_lane_observe = []
    right_lane_observe = []

    # Initialize counters for the current interval
    current_left_lane_count = 0
    current_right_lane_count = 0

    # Set the interval duration

    # Store vehicle delays
    vehicle_delays = {}
    vehicle_travel_times = {}
    vehicle_start_times = {}

    total = 57600
    pbar = tqdm(total)
    # Run the simulation
    step = 0
    start_time = datetime.datetime.now()


    # use detector in python:
    detectors = {
        "detector1": {"lane": "1089324756#1_1", "position": 7.70},
        "detector2": {"lane": "1089324756#1_0", "position": 8.19}
    }

    # Initialize a dictionary to hold vehicle counts
    vehicle_counts = {"detector1": 0, "detector2": 0}

    counted_vehicles = {"detector1": set(), "detector2": set()}

    # Run the simulation
    while step < total:
        control.simulationStep()  # Advance the simulation

        # Detecting vehicles...
        for detector_id, detector_info in detectors.items():
            vehicles_on_lane = control.lane.getLastStepVehicleIDs(detector_info["lane"])
            
            for vehicle in vehicles_on_lane:
                pos = control.vehicle.getLanePosition(vehicle)
                if pos >= detector_info["position"] and pos < detector_info["position"] + 10:

                    # avoid when traffic is jammed, multiple counts over one vechile 
                    if vehicle not in counted_vehicles[detector_id]:
                    
                        if detector_id == "detector1":
                            current_left_lane_count += 1
                        elif detector_id == "detector2":
                            current_right_lane_count += 1
                        
                        # Mark the vehicle as counted
                        counted_vehicles[detector_id].add(vehicle)

        vehicle_ids = control.vehicle.getIDList()
        
        for v_id in vehicle_ids:
            # Get the vehicle's current delay
            delay = control.vehicle.getAccumulatedWaitingTime(v_id)
            
            # Update the total delay
            vehicle_delays[v_id] = delay
        
            # Check for vehicles that have arrived
        for v_id in list(vehicle_start_times):
            if v_id not in control.vehicle.getIDList():
                # Vehicle has finished its route and is no longer in the simulation
                arrival_time = control.simulation.getTime()
                travel_time = arrival_time - vehicle_start_times[v_id]
                vehicle_travel_times[v_id] = travel_time
                # Remove the vehicle from the start times to avoid double counting
                del vehicle_start_times[v_id]
            else:
                if not control.vehicle.isRouteValid(v_id) or control.vehicle.getRoadID(v_id) == '':
                    arrival_time = control.simulation.getTime()
                    travel_time = arrival_time - vehicle_start_times[v_id]
                    vehicle_travel_times[v_id] = travel_time
                    # Remove the vehicle from the start times to avoid double counting
                    del vehicle_start_times[v_id]

            # Mark the departure time for each vehicle
            if v_id not in vehicle_start_times:
                vehicle_start_times[v_id] = control.simulation.getTime()  # libsumo returns time in seconds

        # Check if the current interval has ended
        if step % interval_duration == 0 and step != 0:
            # Store the counts in the respective lists and reset counters
            left_lane_observe.append(current_left_lane_count)
            right_lane_observe.append(current_right_lane_count)
            current_left_lane_count = 0
            current_right_lane_count = 0

        # Your existing vehicle delay and travel time calculation code...

        step += 1
        pbar.update(1)
    


    # Include counts for the last interval if it's not exactly 3600 seconds
    if step % interval_duration != 0:
        left_lane_observe.append(current_left_lane_count)
        right_lane_observe.append(current_right_lane_count)
    
    # Calculate the average delay
    if vehicle_delays:
        average_delay = sum(vehicle_delays.values()) / len(vehicle_delays)
    else: 
        average_delay = 0

    # Calculate the average travel time
    if vehicle_travel_times:
        average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
    else:
        average_travel_time = 0
    # Your existing code to calculate average delay and travel time...

    end_time = datetime.datetime.now()


    # Output the result
    print("Left Lane Observations:", left_lane_observe)
    print("Right Lane Observations:", right_lane_observe)
        # Output the result
    print(f"Average delay time: {average_delay} seconds")
    print(f"Average travel time: {average_travel_time} seconds")
    print(f"total time cost:{end_time - start_time}")
    

    # real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
    real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])
    x_data = np.arange(6, 21)

    plt.xlabel("time/h")
    plt.ylabel("vehicle count")
    plt.title("Simulation info across time on SUMO - Sedona")
    plt.plot(x_data, real_data, label="real", color = 'r')
    plt.plot(x_data, left_lane_observe, label="left_lane")
    plt.plot(x_data, right_lane_observe, label="right_lane")
    plt.plot(x_data, np.array(left_lane_observe)+np.array(right_lane_observe), label="total pass count")
    plt.legend()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if save:
        
        plt.savefig(save_path + "images/"+ str(para_set)+"_"+curr_time, dpi=700)
        plt.clf()
    if viusalize:
        plt.show()
        plt.clf()
        
    return left_lane_observe, right_lane_observe


# def simulate_for_rulelight(save_file=False, viusalize = True, save_path=".", para_set=0):

#     viusalize = viusalize
#     save = save_file
#     controller = "libsumo"
#     interval_duration = 3600

#     # Define the path to your configuration file
#     sumoConfig = "./demand_info/sedona.sumocfg"

#     if controller == "libsumo":
#         control = libsumo
#         control.start(["-c", sumoConfig])
#     elif controller == "traci":
#         control = traci
#         sumoBinary = sumolib.checkBinary('sumo')
#         control.start([sumoBinary, "-c", sumoConfig])

#     # Your existing setup code...

#     # Initialize lists to hold the observation counts for each lane
#     left_lane_observe = []
#     right_lane_observe = []

#     # Initialize counters for the current interval
#     current_left_lane_count = 0
#     current_right_lane_count = 0

#     # Set the interval duration

#     # Store vehicle delays
#     vehicle_delays = {}
#     vehicle_travel_times = {}
#     vehicle_start_times = {}

#     total = 57600
#     pbar = tqdm(total)
#     # Run the simulation
#     step = 0
#     start_time = datetime.datetime.now()


#     # use detector in python:
#     detectors = {
#         "detector1": {"lane": "1089324756#1_1", "position": 7.70},
#         "detector2": {"lane": "1089324756#1_0", "position": 8.19}
#     }

#     # Initialize a dictionary to hold vehicle counts
#     counted_vehicles = {"detector1": set(), "detector2": set()}

#     # Run the simulation
#     while step < total:
#         control.simulationStep()  # Advance the simulation
#         control_traffic_lights(control=control)

#         # Detecting vehicles...
#         for detector_id, detector_info in detectors.items():
#             vehicles_on_lane = control.lane.getLastStepVehicleIDs(detector_info["lane"])
            
#             for vehicle in vehicles_on_lane:
#                 pos = control.vehicle.getLanePosition(vehicle)
#                 if pos >= detector_info["position"] and pos < detector_info["position"] + 10:

#                     # avoid when traffic is jammed, multiple counts over one vechile 
#                     if vehicle not in counted_vehicles[detector_id]:
                    
#                         if detector_id == "detector1":
#                             current_left_lane_count += 1
#                         elif detector_id == "detector2":
#                             current_right_lane_count += 1
                        
#                         # Mark the vehicle as counted
#                         counted_vehicles[detector_id].add(vehicle)

#         vehicle_ids = control.vehicle.getIDList()
        
#         for v_id in vehicle_ids:
#             # Get the vehicle's current delay
#             delay = control.vehicle.getAccumulatedWaitingTime(v_id)
            
#             # Update the total delay
#             vehicle_delays[v_id] = delay

#             # Mark the departure time for each vehicle
#             if v_id not in vehicle_start_times:
#                 vehicle_start_times[v_id] = control.simulation.getTime()  # libsumo returns time in seconds
            
#         # Check for vehicles that have arrived
#         for v_id in list(vehicle_start_times):
#             if v_id not in control.vehicle.getIDList():
#                 # Vehicle has finished its route and is no longer in the simulation
#                 arrival_time = control.simulation.getTime()
#                 travel_time = arrival_time - vehicle_start_times[v_id]
#                 vehicle_travel_times[v_id] = travel_time
#                 # Remove the vehicle from the start times to avoid double counting
#                 del vehicle_start_times[v_id]
#             else:
#                 if not control.vehicle.isRouteValid(v_id) or control.vehicle.getRoadID(v_id) == '':
#                     arrival_time = control.simulation.getTime()
#                     travel_time = arrival_time - vehicle_start_times[v_id]
#                     vehicle_travel_times[v_id] = travel_time
#                     # Remove the vehicle from the start times to avoid double counting
#                     del vehicle_start_times[v_id]

#             # Mark the departure time for each vehicle
#             if v_id not in vehicle_start_times:
#                 vehicle_start_times[v_id] = control.simulation.getTime()  # libsumo returns time in seconds

#         # Check if the current interval has ended
#         if step % interval_duration == 0 and step != 0:
#             # Store the counts in the respective lists and reset counters
#             left_lane_observe.append(current_left_lane_count)
#             right_lane_observe.append(current_right_lane_count)
#             current_left_lane_count = 0
#             current_right_lane_count = 0

#         # Your existing vehicle delay and travel time calculation code...

#         step += 1
#         pbar.update(1)
    
#     # Include counts for the last interval if it's not exactly 3600 seconds
#     if step % interval_duration != 0:
#         left_lane_observe.append(current_left_lane_count)
#         right_lane_observe.append(current_right_lane_count)
    
#     # Calculate the average delay
#     if vehicle_delays:
#         average_delay = sum(vehicle_delays.values()) / len(vehicle_delays)
#     else: 
#         average_delay = 0

#     # Calculate the average travel time
#     if vehicle_travel_times:
#         average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
#     else:
#         average_travel_time = 0
#     # Your existing code to calculate average delay and travel time...

#     end_time = datetime.datetime.now()


#     # Output the result
#     print("Left Lane Observations:", left_lane_observe)
#     print("Right Lane Observations:", right_lane_observe)
#         # Output the result
#     print(f"Average delay time: {average_delay} seconds")
#     print(f"Average travel time: {average_travel_time} seconds")
#     print(f"total time cost:{end_time - start_time}")
    

#     # real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255, 155])
#     real_data = np.array([162, 331, 495, 683, 778, 826, 850, 866, 854, 823, 694, 585, 510, 386, 255])
#     x_data = np.arange(6, 21)

#     plt.xlabel("time/h")
#     plt.ylabel("vehicle count")
#     plt.title("Simulation info across time on SUMO - Sedona")
#     plt.plot(x_data, real_data, label="real", color = 'r')
#     plt.plot(x_data, left_lane_observe, label="left_lane")
#     plt.plot(x_data, right_lane_observe, label="right_lane")
#     plt.plot(x_data, np.array(left_lane_observe)+np.array(right_lane_observe), label="total pass count")
#     plt.legend()
#     curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     if save:
#         plt.savefig(save_path + "images/"+ str(para_set)+"_"+curr_time, dpi=700)
#         plt.clf()
#     if viusalize:
#         plt.show()
#         plt.clf()
        
#     return left_lane_observe, right_lane_observe