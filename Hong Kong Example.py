# This python code is to convert the hospital geo coordinate into distance matrix

import math
import csv

# Function to calculate distance using the Haversine formula
def haversine_distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Input demand points
num_demand_points = int(input("Enter the number of demand points (hospitals): "))
demand_points = {}
for i in range(1, num_demand_points + 1):
    lat = float(input(f"Enter latitude for demand point {i}: "))
    lon = float(input(f"Enter longitude for demand point {i}: "))
    demand_points[i] = (lat, lon)

# Input supply points
num_supply_points = int(input("Enter the number of supply points (hospitals): "))
supply_points = {}
for j in range(num_supply_points):
    lat = float(input(f"Enter latitude for supply point {chr(97 + j)}: "))
    lon = float(input(f"Enter longitude for supply point {chr(97 + j)}: "))
    supply_points[chr(97 + j)] = (lat, lon)

# Create the distance matrix
distance_matrix = [["d_ij"] + [f"Supply {chr(97 + j)}" for j in range(num_supply_points)]]
for i in range(1, num_demand_points + 1):
    row = [f"Demand {i}"]
    for j in range(num_supply_points):
        supply_id = chr(97 + j)
        d = haversine_distance(demand_points[i], supply_points[supply_id])
        row.append(round(d, 2))  # Round to 2 decimal places
    distance_matrix.append(row)

# Print the distance matrix
print("\nDistance Matrix (in km):")
for row in distance_matrix:
    print("\t".join(map(str, row)))

# Save the matrix to a CSV file
with open("hospital_distance_matrix.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(distance_matrix)

print("\nDistance matrix has been saved to 'hospital_distance_matrix.csv'.")
