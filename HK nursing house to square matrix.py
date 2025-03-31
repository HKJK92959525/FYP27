# This python code can read the CSV file (abcd.csv) and convert to CSV square distance martix

import math
import csv

# Function to calculate distance using the Haversine formula
def haversine_distance(coord1, coord2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula calculations
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Specify the input CSV file (make sure it has a header row with at least "hospital name", "latitude", and "longitude")
input_filename = "abcd.csv"

# Read the hospital data from the CSV file
hospitals = []  # Each element will be a tuple: (hospital_name, (latitude, longitude))
with open(input_filename, "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header row
    for row in reader:
        # Assuming the CSV columns are: hospital name, latitude, longitude
        hospital_name = row[0]
        lat = float(row[1])
        lon = float(row[2])
        hospitals.append((hospital_name, (lat, lon)))

# Total number of hospitals
n = len(hospitals)

# Create the square distance matrix with hospital names as labels
# First row: header row with "d_ij" then each hospital name as a column header.
distance_matrix = [["d_ij"] + [hospitals[j][0] for j in range(n)]]

# Also, store the matrix as a Python list of lists (without the header)
distance_list = []

for i in range(n):
    # The first element in each row is the hospital name (row header)
    row = [hospitals[i][0]]
    list_row = []  # For the numeric distance values only
    for j in range(n):
        if i == j:
            d = 0  # Distance to itself is 0
        else:
            d = haversine_distance(hospitals[i][1], hospitals[j][1])
        # Round the result to two decimal places
        d = round(d, 2)
        row.append(d)
        list_row.append(d)
    distance_matrix.append(row)
    distance_list.append(list_row)

# Print the distance matrix (with headers) to the console
print("\nSquare Distance Matrix (in km):")
for row in distance_matrix:
    print("\t".join(map(str, row)))

# Optionally, print the Python list format of the matrix (without headers)
print("\nDistance Matrix in Python List Format:")
print(distance_list)

# Write the distance matrix to a CSV file with UTF-8 encoding
output_filename = "hospital_distance_matrix.csv"
with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(distance_matrix)

print(f"\nSquare distance matrix has been saved to '{output_filename}'.")
