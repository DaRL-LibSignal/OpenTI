import requests

# Define the latitude and longitude range

# Define your API key
api_key = 'AIzaSyBzjjW_1gTjWmqkdekafOR1mQUfwh8uf7g'


location = '[-74.0003257, 40.7258656, -73.9914629, 40.7323333]'

print(location)

# Evaluating the string to convert it into a list of floats
target = eval(location)

# Correctly assigning min and max latitudes and longitudes
min_long, min_lat, max_long, max_lat = target[0], target[1], target[2], target[3]

# Calculate the center of the bounding box
center_lat = (min_lat + max_lat) / 2
center_long = (min_long + max_long) / 2
center = f'{center_lat},{center_long}'

# Define a zoom level (adjust as necessary)
zoom = 15  # You might need to adjust this value based on the area covered by your bounding box

# Define the URL for the Static Map API
url = f'https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size=600x400&key={api_key}'

# Make the request and get the response
response = requests.get(url)

# Save the image to a file
if response.status_code == 200:
    with open('/home/local/ASURITE/longchao/Desktop/project/LLM4Traffic/OpenTI/pivotalAgent/AugmentUtils/map.png', 'wb') as file:
        file.write(response.content)
else:
    print(f"Failed to get the map. Status code: {response.status_code}")

# Print the URL for reference
print(f"Map URL: {url}")