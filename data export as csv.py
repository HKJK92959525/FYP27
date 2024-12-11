import math
import csv

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Get the number of demand points from the user
num_demand_points = int(input("Enter the number of demand points: "))

# Collect the demand points from the user
demand_points = {}
for i in range(1, num_demand_points + 1):
    x = float(input(f"Enter x-coordinate for demand point {i}: "))
    y = float(input(f"Enter y-coordinate for demand point {i}: "))
    demand_points[i] = (x, y)

# Get the number of supply points from the user
num_supply_points = int(input("Enter the number of supply points: "))

# Collect the supply points from the user
supply_points = {}
for j in range(num_supply_points):
    x = float(input(f"Enter x-coordinate for supply point {chr(97 + j)}: "))  # chr(97) gives 'a'
    y = float(input(f"Enter y-coordinate for supply point {chr(97 + j)}: "))
    supply_points[chr(97 + j)] = (x, y)

# Create the distance matrix
distance_matrix = [["d_ij"] + [f"Supply {chr(97 + j)}" for j in range(num_supply_points)]]
for i in range(1, num_demand_points + 1):
    row = [f"Demand {i}"]
    for j in range(num_supply_points):
        supply_id = chr(97 + j)  # Map to 'a', 'b', 'c', etc.
        d = calculate_distance(demand_points[i], supply_points[supply_id])
        row.append(round(d, 2))  # Round for readability
    distance_matrix.append(row)

# Print the matrix
print("\nDistance Matrix:")
for row in distance_matrix:
    print("\t".join(map(str, row)))

# Save the matrix to a CSV file
with open("distance_matrix_demand_supply.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(distance_matrix)

print("\nDistance matrix has been saved to 'distance_matrix_demand_supply.csv'.")
