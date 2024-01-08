import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import subprocess
import os, sys
import datetime
import argparse
import yaml

"""
This is to simulate by one click

all you need:
[map.osm]
"""

def click_simulate(path=""):

    assert os.path.exists(path), "The file is not existing!"
    
    # create working dir
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    working_dir = "./autoSimuOutput/"+curr_time+"/"
    os.makedirs(working_dir)
    save_path = working_dir + "images/"
    os.makedirs(save_path)

    working_dir = "./autoSimuOutput/" + curr_time + "/"
    sumo_path =  yaml.load(open('../../Configs/path.yaml'), Loader=yaml.FullLoader)['SUMO_PATH']

    # bash commands
    commands = f"""
    export SUMO_HOME={sumo_path}
    export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
    cd {working_dir}
    cp {path} ./
    netconvert --osm-files map.osm -o map.net.xml
    python {sumo_path}/tools/randomTrips.py -n map.net.xml -e 100 -l
    python {sumo_path}/tools/randomTrips.py  -n map.net.xml -r map.rou.xml -e 100 -l
    echo $PWD
    cp ../../../../../assets/base.sumocfg ./
    sumo-gui base.sumocfg
    """

    # Run commands to make sure the sumo env is okay
    subprocess.run(commands, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process expected parameters for simulation!')
    parser.add_argument('-f', dest='fpath', default="", help='your parameter')
    args = parser.parse_args()
    if args.fpath:
        click_simulate(path=args.fpath)

    

