import os
import sys
import copy
import yaml
import time
import random
import requests
import osm2gmns as og
import subprocess, re
from DLSim import DLSim
from .painter import painter
from grid2demand import GRID2DEMAND
import datetime
import matplotlib
import folium
from folium.plugins import FloatImage
import urllib.request
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
# map html-style visualization
import urllib.request
from PIL import Image
import os
import math
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

config_path = "./pivotalAgent/Configs/path.yaml"

with open(config_path, 'r') as file:
    tsc_root = yaml.safe_load(file)["LibSignal"]["tsc_root"]

def func_prompt(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

import json


class ask4Area:
    def __init__(self) -> None:
        pass

    @func_prompt(name="queryAreaRange",
             description="""
             This tool is used to obtain the area information of a interest point on map.
             Consider using this tool when asked "Where's the interest point?"
             The output will tell you whether you have finished this command successfully. 
             """)

    def embody(self, target: str) -> str:
        # Example input string
        def get_bounding_box(address):
        # URL of the Nominatim API
            nominatim_url = "https://nominatim.openstreetmap.org/search"

            # Parameters for the API request
            params = {
                'q': address,  # Query with the provided address
                'format': 'json',  # Response format
                'limit': 1  # Limit to the top result
            }

            # Sending a request to the Nominatim API
            response = requests.get(nominatim_url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data:
                    # Extracting bounding box
                    bounding_box = data[0]['boundingbox']
                    lat_min, lat_max, long_min, long_max= bounding_box
                    return [float(long_min), float(lat_min), float(long_max), float(lat_max)]
                   
                else:
                    return "No data found for this address."
            else:
                return "Failed to retrieve data."
        
        location = get_bounding_box(target)

        try:
            return f"If you successuflly loacated the map. just return You have successfully located the map of: {target}. And your final answer should include this sentence without changing anything: The longitude and latitude area of interested {target} is:{location}."
        except json.JSONDecodeError:
            return "Invalid location format. Please provide a valid list of longitude and latitude values."

class GoogleMapsLayers:
        ROADMAP = "v"
        TERRAIN = "p"
        ALTERED_ROADMAP = "r"
        SATELLITE = "s"
        TERRAIN_ONLY = "t"
        HYBRID = "y"

class GoogleMapDownloader:
        def __init__(self, min_long, min_lat, max_long, max_lat, zoom=12, layer=GoogleMapsLayers.ALTERED_ROADMAP):
            self._west = min_long
            self._east = max_long
            self._south = min_lat
            self._north = max_lat
            self._zoom = zoom
            self._layer = layer

        def getXY(self):
            tile_size = 256
            num_tiles = 1 << self._zoom

            top_left_x = (tile_size / 2 + self._west * tile_size / 360.0) * num_tiles // tile_size
            sin_top_left_y = math.sin(self._north * (math.pi / 180.0))
            top_left_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_top_left_y) / (1 - sin_top_left_y)) * -(
                tile_size / (2 * math.pi))) * num_tiles // tile_size

            bottom_right_x = (tile_size / 2 + self._east * tile_size / 360.0) * num_tiles // tile_size
            sin_bottom_right_y = math.sin(self._south * (math.pi / 180.0))
            bottom_right_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_bottom_right_y) / (1 - sin_bottom_right_y)) * -(
                tile_size / (2 * math.pi))) * num_tiles // tile_size

            return int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)

        def generateImage(self, **kwargs):
            start_x = kwargs.get('start_x', None)
            start_y = kwargs.get('start_y', None)
            tile_width = kwargs.get('tile_width', 5)
            tile_height = kwargs.get('tile_height', 5)

            if start_x is None or start_y is None:
                start_x, start_y, _, _ = self.getXY()

            width, height = 256 * tile_width, 256 * tile_height
            map_img = Image.new('RGB', (width, height))

            for x in range(0, tile_width):
                for y in range(0, tile_height):
                    url = f'https://mt0.google.com/vt?lyrs={self._layer}&x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                        self._zoom)

                    current_tile = str(x) + '-' + str(y)
                    urllib.request.urlretrieve(url, current_tile)

                    im = Image.open(current_tile)
                    map_img.paste(im, (x * 256, y * 256))

                    os.remove(current_tile)

            return map_img 


class GoogleMapsLayers:
        ROADMAP = "v"
        TERRAIN = "p"
        ALTERED_ROADMAP = "r"
        SATELLITE = "s"
        TERRAIN_ONLY = "t"
        HYBRID = "y"

class GoogleMapDownloader:
        def __init__(self, min_longtitude, min_latitude, max_longtitude, max_latitude, zoom=12, layer=GoogleMapsLayers.ALTERED_ROADMAP):
            self._west = min_longtitude
            self._east = max_longtitude
            self._south = min_latitude
            self._north = max_latitude
            self._zoom = zoom
            self._layer = layer

        def getXY(self):
            tile_size = 256
            num_tiles = 1 << self._zoom

            top_left_x = (tile_size / 2 + self._west * tile_size / 360.0) * num_tiles // tile_size
            sin_top_left_y = math.sin(self._north * (math.pi / 180.0))
            top_left_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_top_left_y) / (1 - sin_top_left_y)) * -(
                tile_size / (2 * math.pi))) * num_tiles // tile_size

            bottom_right_x = (tile_size / 2 + self._east * tile_size / 360.0) * num_tiles // tile_size
            sin_bottom_right_y = math.sin(self._south * (math.pi / 180.0))
            bottom_right_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_bottom_right_y) / (1 - sin_bottom_right_y)) * -(
                tile_size / (2 * math.pi))) * num_tiles // tile_size

            return int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)

        def generateImage(self, **kwargs):
            start_x = kwargs.get('start_x', None)
            start_y = kwargs.get('start_y', None)
            tile_width = kwargs.get('tile_width', 5)
            tile_height = kwargs.get('tile_height', 5)

            if start_x is None or start_y is None:
                start_x, start_y, _, _ = self.getXY()

            width, height = 256 * tile_width, 256 * tile_height
            map_img = Image.new('RGB', (width, height))

            for x in range(0, tile_width):
                for y in range(0, tile_height):
                    url = f'https://mt0.google.com/vt?lyrs={self._layer}&x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                        self._zoom)

                    current_tile = str(x) + '-' + str(y)
                    urllib.request.urlretrieve(url, current_tile)

                    im = Image.open(current_tile)
                    map_img.paste(im, (x * 256, y * 256))

                    os.remove(current_tile)

            return map_img 

class showMap:
    def __init__(self) -> None:
        pass 
        # Specify the bounding box coordinates

    @func_prompt(name="showOnMap",
            description="""
             This tool is used to show the interested loaction on Map, like the ASU campus area on map.
             Consider using this tool when asked "Can you show me this area on the city map ?"
             The output will tell you whether you have finished this command successfully.
             """)
    def embody(self, target: str) -> str:
        target = eval(target)
        min_long,min_lat,max_long,max_lat = target[0],target[1],target[2],target[3]
        gmd = GoogleMapDownloader(min_long, min_lat, max_long, max_lat, 13, GoogleMapsLayers.ALTERED_ROADMAP)

        def create_and_show_map(bounding_box):
            min_lon,min_lat,max_lon,max_lat = bounding_box[0], bounding_box[1],bounding_box[2],bounding_box[3]
            # Center of the map
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Create a map object
            map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=13)

            # Add bounding box as a rectangle
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color='#ff7800',
                fill=True,
                fill_color='#ffff00',
                fill_opacity=0.2
            ).add_to(map_obj)

            return map_obj
        try:
            img = gmd.generateImage()
        except:
            return "Error for finding the place"
        else:
            img.save("Data/maps/target_map.png")
            print("Create map Successfully")
        # LLMAgent/simulation/asu_map.png 
        map = create_and_show_map(target)
        html_path = "Data/maps/target_map.html"
        map.save("Data/maps/target_map.html")

        target_map ="./Data/maps/target_map.png"
        
        return f"You have successfully find the map of: {target}. And your final answer should include this sentence without changing anything: The map area of interested {target} is: `{target_map}`, and dynamic version please check the link above:`{html_path}`."

class autoDownloadNetwork:
    def __init__(self, base_loc: str) -> None:
        self.base_loc = base_loc
    
    @func_prompt(name="autoDownloadOpenStreetMapFile",
             description="""
             This tool is used to automatically download the certain area's map data from OpenStreetMap. The downloaded file is data ends with '.osm'.
             This tool will return the file path that the downloaded file has been stored. 
             Consider using this tool if the question is anything about get me the map of area, or download the map from openstreetmap, or I want to download the map data from openstreetmap(OpenStreetMap).
             The output will tell you whether you have finished this command successfully.
             """
             )

    def embody(self, target: str) -> str:

        try: 
            print("get the data in download: {}".format(target))
            desired = target.replace(" ", "").split(",")[-1].replace('"', '').replace("[", "").replace("]", "")
            print("desired:"+desired)
            long_min, lat_min, long_max, lat_max = target.replace("[", "").replace("]", "").strip().replace(" ", "").split(",")[:4]
            print("long_min, lat_min, long_max, lat_max")
            print(long_min, lat_min, long_max, lat_max)
            url = "https://www.openstreetmap.org/api/0.6/map?bbox={}%2C{}%2C{}%2C{}".format(long_min, lat_min, long_max, lat_max)
            print("url:")
            print(url)
            response = requests.get(url)
            if response.status_code == 200:              
                file_path = self.base_loc+ desired
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                if not os.path.exists(file_path):
                    return f"The requested cannot be successfully downloaded."
            else:
                file_path = 'None'
                print("Failed to retrieve the data.")
                return f"The requested cannot be successfully downloaded."
           
            time.sleep(2)
            return f"The requested have successfully downloaded and saved at: {file_path}. And your final answer should include this sentence without changing anything: The file saved location is at: `{file_path}`."
        except FileNotFoundError as e:
            return f"The requested cannot be successfully downloaded because your request was too large. Either request a smaller area, or use planet.osm."

def extract_filepath(s):
    match = re.search(r'data/.*\.log', s)
    return match.group(0) if match else None

def update_episode_in_base_yaml(episode_value: str) -> None:
    base_yaml_path = "./LibSignal/LibSignal/configs/tsc/base.yml"  # Replace with the actual path to your base.yml file

    with open(base_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the episodes value
    data["trainer"]["episodes"] = int(episode_value)

    with open(base_yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

class simulateOnLibSignal:
    def __init__(self, base_dir: str, save_dir: str) -> None:
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.temp_directfeedback = tsc_root
    
    @func_prompt(name="simulateOnLibsignal",
             description="""
             This tool is used to execute the simulation on open source library called LibSignal. 
             Consider using this tool when it is mentioned of LibSignal and run/execution/simulate in/on LibSignal.
             The output will tell you whether you have finished this command successfully.
             """)
    
    
    def embody(self, target: str) -> str:
        print("target:")
        print(target)

        support_simulators = ["cityflow", "sumo"]
        support_algorithms_1 = ["dqn", "frap", "presslight", "mplight"]
        support_algorithms_2 = ["fixedtime", "sotl"]
        # try:
        print("get the command:{}".format(target))


        # Using regular expressions to extract information
        matches = re.match(r'([^,]+),\s*([^,]+),\s*(\d+)', target)

        if matches:
            simulator = matches.group(1).strip()
            algorithm = matches.group(2).strip()
            episode = int(matches.group(3).strip())
            print(simulator, algorithm, episode)
        else:
            # Using string manipulation to extract information
            simulator = None
            algorithm = None
            episode = None

            for param in target.replace(" ", "").split(","):
                if param.startswith("simulator="):
                    simulator = param.split("=")[1]
                elif param.startswith("algorithm="):
                    algorithm = param.split("=")[1]
                elif param.startswith("episode="):
                    episode = int(param.split("=")[1])

        print(simulator, algorithm, episode)
        
       
            
        # Set default value if episode is not provided
        episode = int(episode) if episode is not None else 5
        
        # Call the function to update the episode in base.yml
        update_episode_in_base_yaml(episode)

        
        with open(config_path, 'r') as file:
            log_example = yaml.safe_load(file)["LibSignal"]["log_example"]
            
        with open(config_path, 'r') as file2:
            libsignal_root = yaml.safe_load(file2)["LibSignal"]["root_path"]

        print(simulator, algorithm, episode)
        if simulator == "cityflow":
            if algorithm.lower() in support_algorithms_1:

           
                base_path = libsignal_root
                
                # Generate a timestamp for the log file
                timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                
                # Define the relative path for the log file
                log_file_relative_path = "data/output_data/tsc/cityflow_{}/cityflow1x1/0/logger/{}_DTL.log".format(algorithm,timestamp)
                
                # Combine the base path and relative path to get the full log file path
                log_file_path = os.path.join(base_path, log_file_relative_path)
                completed_process = subprocess.run(["python3", base_path+"run.py", "-a", algorithm,"-tt", timestamp], capture_output=True, text=True, cwd=base_path)
                # Check if the subprocess completed successfully
                output = completed_process.stdout
                print("output:")
                print(output)
                
                read_path = log_file_path

                saved_image = painter({'hz1x1': read_path}, ['epoch', 'average travel time', 'rewards','delay'])
                return f"Your final answer should include this sentence without changing anything: The simulation results are saved at:`{saved_image}` and the log file is saved at:{log_file_path}."
                
            
            
            elif algorithm.lower() in support_algorithms_2:
                timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                
                subprocess.run(["python3", self.base_dir+"run.py", "-a", algorithm,"-tt", timestamp], capture_output=True, text=True, cwd=libsignal_root)

                sub_half = simulator + "_" + algorithm + "/cityflow1x1/0/logger/example_BRF.log"
                path = self.temp_directfeedback + sub_half

                with open(file=path, mode="r") as reader:
                    simulation_result = reader.read()                 
                return f"Your final answer should include this sentence without changing anything: The simulation results are: {simulation_result}."
        

class filterNetwork:
    def __init__(self, figfolder: str) -> None:
        # base_network: 
        self.figfolder = figfolder
        self.store_base = "./Data/netfilter/output/"
    
    # The input most likely contains the information related to keywords like: filter network, walk/walkable, bike/bikeable/, railway/railway routes.
    @func_prompt(name='networkFilter',
             description="""
            This tool is used to filter the network by the required categories.
            This tool will also return the file path of a filtered road network with emphasized lanes of interest to provide the final answer. 
            Consider Using this tool if the question is about filtering or obtaining a walkable/bikeable/railway/ network from the base network.
            The output will tell you whether you have finished this command successfully.
             """)

    def embody(self, target: str) -> str:
        try:
            # get the target information:
            print("target:")
            print(target)
            target = target.replace("[", "").replace("]", "").replace(" ", "")

            target_path, keyword = target.split(",")
            time.sleep(0.5)
            try:
                net = og.getNetFromFile(target_path, network_types=keyword)
            except Exception as e :
                print("There are some error when filtering")

            time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            store_info = self.store_base + time_now + "-" + keyword +".png"

            og.osmnet.visualization.saveFig(network=net, picpath=store_info)
            if not os.path.exists(store_info):
                return f"You cannot successfully filter the network."
            
            return f"You have successfully filter the network by type: {target} on the target network. And your final answer should include this sentence without changing anything except for translation: The location of interested {target} is kept at: `{store_info}`."
        except Exception as e:
                return f"You cannot  filter the nework by type: {target} on the target network."

class generateDemand:

    def __init__(self, save: str) -> None:
        self.demand_save = save
    
    @func_prompt(name="generateDemand", 
             description="""This tool is used for generating demand based on the osm data, it leverages the package named "grid2demand".
             Consider using this tool when asked to generate traffic demand based on the downloaded map/osm file. And you will need the a string of path to the .osm file.
             The output will tell you whether you have finished this command successfully.
             """)
    
    def embody(self, target: str) -> str:
        import grid2demand as gd

        print("demand target:{}".format(target))
        name = target.replace(".osm", "").split("/")[-1]
        dump_dir = "./Data/demand/"+name
        net = og.getNetFromFile(target, network_types=('walk', 'auto') , POI=True, default_lanes=True, default_speed=True)    
        og.connectPOIWithNet(net)
        og.generateNodeActivityInfo(net)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        og.outputNetToCSV(net, output_folder=dump_dir)
        gd.read_network(dump_dir)

        # Initialize a GRID2DEMAND object
        gd = GRID2DEMAND(dump_dir)

        # Load node and poi data from input directory
        node_dict, poi_dict = gd.load_network.values()

        # Generate zone dictionary from node dictionary by specifying number of x blocks and y blocks
        zone_dict = gd.net2zone(node_dict, num_x_blocks=20, num_y_blocks=20)

        # synchronize geometry info between zone, node and poi
        # add zone_id to node and poi dictionaries
        # also add node_list and poi_list to zone dictionary
        updated_dict = gd.sync_geometry_between_zone_and_node_poi(zone_dict, node_dict, poi_dict)
        zone_dict_update, node_dict_update, poi_dict_update = updated_dict.values()

        # Calculate zone-to-zone od distance matrix
        zone_od_distance_matrix = gd.calc_zone_od_distance_matrix(zone_dict_update)

        # Generate poi trip rate for each poi
        poi_trip_rate = gd.gen_poi_trip_rate(poi_dict_update)

        # Generate node production attraction for each node based on poi_trip_rate
        node_prod_attr = gd.gen_node_prod_attr(node_dict_update, poi_trip_rate)

        #Calculate zone production and attraction based on node production and attraction
        zone_prod_attr = gd.calc_zone_prod_attr(node_prod_attr, zone_dict_update)

        #Run gravity model to generate agent-based demand
        df_demand = gd.run_gravity_model(zone_prod_attr, zone_od_distance_matrix)

        # Generate agent-based demand
        df_agent = gd.gen_agent_based_demand(node_prod_attr, zone_prod_attr, df_demand=df_demand)

        # You can also view and edit the package setting by using gd.pkg_settings
        print(gd.pkg_settings)

        #Output demand, agent, zone, zone_od_dist_table, zone_od_dist_matrix files to output directory
        gd.save_demand
        df_demand.to_csv(dump_dir+"/demand2.csv")
        gd.save_agent
        gd.save_zone
        gd.save_zone_od_dist_table
        gd.save_zone_od_dist_matrix

        return f"You have successfully generated the demand files. And your final answer should include this sentence without changing anything except for translation: The generated demand is kept at: {dump_dir}."


class simulateOnDLSim:
    def __init__(self, demand_path: str) -> None:
        self.demand_path = demand_path
        self.simulate_path = "./AugmentUtils/simulation/"

    @func_prompt(name="simulateOnDLSim", description="""
    This tool is used for simulating on the DLSim multi-resolution traffic simulator. 
    Please consider using this tool when asked to run simulation on DLSim simulator given a demand path.
    """)

    def embody(self, target: str) -> str:
        target = "./AugmentUtils/simulation/simulate/"
        print("DLSim target:{}".format(target))
        try: 
            subprocess.run(["python3", self.simulate_path+"simulate.py"])
        except Exception as e:
            print(e)
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        base_path = "./AugmentUtils/simulation/simulate/"

        save_to = base_path + time_now + "-log.txt"

        log_file = base_path + "log.txt"
        with open(file=log_file, mode="r") as content:
            log_data = content.read()
            keep = log_data.split("Step 1.7:")[-1]
            output = "Step 1.7:" + keep
        
        with open(file=save_to, mode="w+") as new_file: 
            new_file.write(log_data)

        print("output:")
        print(output)

        return f"You have successfully simulated on DLSim. And your final answer should include this sentence without changing anything: The simulation process and logs are saved below: `{save_to}`."
    

class visualizeDemand:
    def __init__(self, demand_path: str) -> None:
        self.demand_path = demand_path

    @func_prompt(name="visualizeDemand", description="""
    This tool is used for visualizing the demand file generated automatically. 
    Please consider using this tool when asked to present a visualization on the generated demand file.
    The output will tell you whether you have finished this command successfully.
    """)

    def embody(self, target: str) -> str:

        print("DLSim target:{}".format(target))

        with open(config_path, 'r') as file:
            example_root = yaml.safe_load(file)["Demand"]["example"]
        
        base_path = example_root

        return f"You have successfully visualized the traffic demand information. And your final answer should include this sentence without changing anything: The traffic demand information at ASUtempe as below: `{base_path}`."
    

class log_analyzer:
    def __init__(self) -> None:
        pass
    
    @func_prompt(name="logAnalyzer", description="""
    This is the tool used to analyze the log files and provide comparison, if you are asked to analyze any given loaction files, please try your best to find the data, and provide logical and rational understanding on it.
    The output will tell you whether you have finished this command successfully.
    """)

    def embody(self, target: str):
        try:
            #process the input target information and provide the acceptable form
            path = target 
            with open(file=path, mode="r") as content:
                log_data = content.read()
            return f"You have the content read now, please provide your own understanding and provide the explained result of it, the content is: "+ log_data
        except Exception as e:
            return "Your final answer should include this sentence without changing anything: The path you provided is not valid, please examine again or it is also acceptable to pass a absolute path, thank you!"
        

class oneClickSUMO:
    def __init__(self) -> None:
        self.pipline_path = "/pivotalAgent/AugmentUtils/one_click_SUMO/"
    @func_prompt(name="oneClickSUMOSimulation", description="""
    This is the tool used to execute one click run SUMO simulation, if you are asked to run sumo simulation with one click, please consider using this tool. Before using, please check for an osm file path.
    The output will tell you whether you have finished this command successfully.
    """)

    def embody(self, target: str):
        try:
            osm_file_path = target
            import os
            current_path = os.getcwd()

            print("target:")
            print(target)
            print("self.pipline_path")
            print(self.pipline_path)
            print("current path:")
            print(current_path)

            if osm_file_path == None: return "Your final answer should include this sentence without changing anything: You are expected to input your osm file path (absolute path) to run this simulation!"
            subprocess.run(["python3", current_path+ self.pipline_path + "simulate_pipline.py", "-f", target], capture_output=True, text=True, cwd=current_path+ self.pipline_path)
            return "Successfully executed SUMO simulation with random sampled demand information."
        
        except Exception as e:
            return "Your final answer should include this sentence without changing anything: The execution happened with error, please examine the path or format, thank you!"
        