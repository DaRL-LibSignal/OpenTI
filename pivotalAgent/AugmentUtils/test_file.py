
import requests
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



# The longitude and latitude area of interested New York University is: ['40.7258656', '40.7323333', '-74.0003257', '-73.9914629'].

# /home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Data/subArea/scenario1/sunCityWest.osm
# if os.path.exists(target):

# min-max lat, min max long
        # ['40.7258656', '40.7323333', '-74.0003257', '-73.9914629']
# location = '[40.7258656, 40.7323333, -74.0003257, -73.9914629]'
        
location = '[-74.0003257, 40.7258656, -73.9914629, 40.7323333]'

print(location)

# Evaluating the string to convert it into a list of floats
target = eval(location)

# Correctly assigning min and max latitudes and longitudes
min_long, min_lat, max_long, max_lat = target[0], target[1], target[2], target[3]

print(f"Min Longitude: {min_long}, Min Latitude: {min_lat}, Max Longitude: {max_long}, Max Latitude: {max_lat}")

# Assuming GoogleMapDownloader and GoogleMapsLayers are defined elsewhere
gmd = GoogleMapDownloader(min_long, min_lat, max_long, max_lat, 13, GoogleMapsLayers.ALTERED_ROADMAP)

def create_and_show_map(bounding_box):
    min_lon, min_lat, max_lon, max_lat = bounding_box[0], bounding_box[1],bounding_box[2],bounding_box[3]
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
    print("Error for finding the place")
else:
    img.save("/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Data/maps/target2_map.png")
    print("Create map Successfully")


# LLMAgent/simulation/asu_map.png 
map = create_and_show_map(target)
html_path = "/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Data/maps/target2_map.html"
map.save("/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Data/maps/target2_map.html")

target_map ="/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/Data/maps/target2_map.png"



