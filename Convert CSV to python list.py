import csv

def csv_to_distance_matrix(file_path):
    distance_matrix = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Exclude the first column (demand label) and convert values to floats
            distance_matrix.append([float(value) for value in row[1:]])
    return distance_matrix

# Example usage
file_path = 'HK nursing house distance matrix.csv'  # Replace with your CSV file path
distance_matrix = csv_to_distance_matrix(file_path)
print("Distance Matrix in Python List Format:")
print(distance_matrix)
