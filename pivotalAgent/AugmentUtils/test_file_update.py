import requests
import json



def get_bounding_box(query):
    # URL for the API request with the provided query
    url = f"https://nominatim.openstreetmap.org/search.php?q={query}&polygon_geojson=1&format=jsonv2"

    # Headers to simulate the request
    headers = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://nominatim.openstreetmap.org/ui/search.html?q=arizona+state+university+',
        'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Linux"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }

    try:
        # Make the GET request with headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Assuming we need the bounding box information from the first result
            if len(data) > 0:
                bounding_box = data[0].get('boundingbox', [])
                if bounding_box:
                    return bounding_box
                else:
                    print("Bounding box not found in the response.")
            else:
                print("No results found in the response.")
        else:
            print(f"Failed to get a response. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def print_bounding_box(bounding_box):
    if bounding_box:
        print(f"Bounding Box: {bounding_box}")
        print(f"North Latitude: {bounding_box[1]}")
        print(f"South Latitude: {bounding_box[0]}")
        print(f"West Longitude: {bounding_box[2]}")
        print(f"East Longitude: {bounding_box[3]}")
    else:
        print("Bounding box is empty.")

# Define the query for Arizona State University
query = "arizona state university"

# Get the bounding box for the query
bounding_box = get_bounding_box(query)

# Print the bounding box
print_bounding_box(bounding_box)