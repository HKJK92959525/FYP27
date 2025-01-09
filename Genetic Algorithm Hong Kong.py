import numpy as np
import random
import matplotlib.pyplot as plt

I = 28  
J = 15  
M = 150  
C = 50   
T = 9  
V = 1  


dij = [[7.6, 6.07, 6.56, 6.83, 9.62, 11.49, 7.83, 8.27, 9.38, 6.4, 11.69, 13.34, 12.65, 31.2, 30.83], 
       [7.49, 0.26, 2.75, 2.76, 3.36, 5.21, 3.87, 4.3, 8.07, 10.81, 7.57, 8.6, 11.76, 25.54, 25.16], 
       [17.09, 16.83, 15.62, 15.35, 14.52, 11.92, 18.62, 18.7, 22.86, 27.42, 19.29, 18.31, 25.81, 23.2, 22.89], 
       [7.13, 1.34, 3.11, 3.26, 4.9, 6.81, 4.26, 4.74, 7.83, 9.42, 8.25, 9.53, 11.55, 26.88, 26.5], 
       [9.94, 5.54, 5.71, 5.45, 2.77, 0.19, 6.87, 6.99, 11.24, 15.91, 8.38, 8.18, 14.48, 21.95, 21.58], 
       [9.41, 4.83, 5.05, 4.8, 2.27, 0.69, 6.34, 6.49, 10.77, 15.23, 8.15, 8.12, 14.09, 22.46, 22.09], 
       [10.09, 5.62, 5.84, 5.58, 2.75, 0.15, 6.85, 6.96, 11.2, 15.95, 8.28, 8.05, 14.42, 21.8, 21.42], 
       [9.72, 3.39, 4.85, 4.69, 0.22, 2.7, 4.03, 4.19, 8.47, 13.21, 6.19, 6.55, 11.84, 22.54, 22.16], 
       [13.83, 6.77, 9.49, 9.54, 7.12, 9.68, 3.14, 2.9, 1.39, 8.6, 3.73, 5.68, 4.98, 23.57, 23.22], 
       [11.61, 5.5, 7.94, 8.07, 7.43, 9.99, 3.64, 3.84, 3.61, 6.21, 6.5, 8.38, 7.1, 26.43, 26.07], 
       [12.68, 5.54, 8.27, 8.31, 5.96, 8.56, 1.91, 1.74, 2.6, 8.8, 3.73, 5.6, 6.23, 23.65, 23.29], 
       [12.68, 5.46, 8.2, 8.23, 5.72, 8.3, 1.72, 1.48, 2.81, 9.12, 3.46, 5.31, 6.36, 23.37, 23.01], 
       [11.1, 3.74, 6.45, 6.46, 3.99, 6.6, 0.16, 0.63, 4.66, 9.87, 4.16, 5.62, 8.23, 23.48, 23.11], 
       [15.04, 8.15, 10.86, 10.92, 8.61, 11.17, 4.63, 4.39, 0.14, 8.18, 4.58, 6.48, 3.59, 23.96, 23.62], 
       [13.0, 9.23, 10.94, 11.16, 11.99, 14.42, 8.5, 8.72, 7.09, 1.38, 11.12, 13.06, 9.13, 30.99, 30.64], 
       [12.81, 7.38, 9.64, 9.8, 9.49, 12.05, 5.65, 5.79, 4.03, 4.38, 7.96, 9.9, 6.72, 27.83, 27.48], 
       [27.7, 24.55, 24.59, 24.31, 21.13, 19.13, 24.4, 24.23, 27.4, 34.24, 22.88, 21.11, 29.03, 14.63, 14.46], 
       [15.84, 8.59, 10.93, 10.82, 6.24, 7.82, 5.4, 4.95, 6.48, 14.22, 1.92, 0.17, 8.11, 18.15, 17.79], 
       [19.34, 12.08, 14.43, 14.32, 9.66, 10.96, 8.68, 8.21, 8.38, 16.49, 4.71, 3.34, 8.48, 15.64, 15.3], 
       [27.57, 20.35, 23.08, 23.08, 19.43, 21.44, 16.46, 16.04, 12.77, 18.42, 13.33, 13.55, 9.29, 21.38, 21.18], 
       [20.86, 14.2, 16.88, 16.96, 14.45, 16.91, 10.63, 10.35, 6.19, 10.55, 9.12, 10.42, 2.61, 24.87, 24.57], 
       [20.93, 14.29, 16.97, 17.05, 14.57, 17.04, 10.74, 10.46, 6.28, 10.51, 9.26, 10.57, 2.74, 25.02, 24.73], 
       [32.59, 25.19, 27.83, 27.78, 23.52, 25.06, 21.33, 20.86, 18.37, 24.84, 17.61, 17.14, 15.31, 18.31, 18.2], 
       [20.79, 14.08, 16.77, 16.84, 14.28, 16.73, 10.49, 10.19, 6.06, 10.64, 8.9, 10.18, 2.43, 24.62, 24.33], 
       [27.86, 20.65, 23.37, 23.38, 19.72, 21.72, 16.75, 16.33, 13.05, 18.67, 13.62, 13.83, 9.57, 21.46, 21.26], 
       [30.55, 23.52, 25.65, 25.51, 20.64, 21.16, 20.29, 19.82, 19.43, 27.41, 16.32, 14.83, 18.04, 8.03, 7.91], 
       [31.8, 25.66, 27.2, 26.99, 22.23, 21.8, 23.31, 22.9, 23.89, 32.0, 19.82, 17.95, 23.55, 0.14, 0.32], 
       [34.4, 27.63, 29.57, 29.4, 24.51, 24.66, 24.65, 24.19, 24.15, 32.19, 20.76, 19.12, 22.91, 5.88, 6.01]
       ]


wi = [4, 7, 4, 6, 9, 2, 5, 2, 8, 6, 5, 5, 9, 10, 9, 6, 5, 8, 3, 9, 7, 5, 9, 7, 8, 4, 3, 4]


max_distance = V * T


def objective_function(solution):
    selected_facilities = solution  
    total_covered_demand = 0
    
    for i in range(I):
        is_covered = any(
            selected_facilities[j] == 1 and dij[i][j] <= max_distance
            for j in range(J)
        )
        if is_covered:
            total_covered_demand += wi[i]
    return total_covered_demand


def get_coverage_details(solution):
    selected_facilities = solution
    coverage_details = {j: [] for j in range(J) if selected_facilities[j] == 1}  
    
    for i in range(I):
        for j in range(J):
            if selected_facilities[j] == 1 and dij[i][j] <= max_distance:
                coverage_details[j].append(i + 1)  
    
    return coverage_details

def initialize_population(size):
    population = []
    for _ in range(size):
        facilities = [random.choice([0, 1]) for _ in range(J)]
        while sum(facilities) * C > M:
            facilities[random.randint(0, J - 1)] = 0
        population.append(facilities)
    return population

def evaluate_population(population):
    return [objective_function(individual) for individual in population]

def select(population, fitness):
    total_fitness = sum(fitness)
    if total_fitness == 0: 
        probabilities = [1 / len(fitness) for _ in fitness]
    else:
        probabilities = [f / total_fitness for f in fitness]
    
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities, replace=True)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    while sum(individual) * C > M:
        individual[random.randint(0, len(individual) - 1)] = 0
    return individual

def genetic_algorithm(population_size, generations, mutation_rate):
    population = initialize_population(population_size)
    best_fitness_over_time = []
    best_solution = None
    
    for generation in range(generations):
        fitness = evaluate_population(population)
        best_fitness = max(fitness)
        best_fitness_over_time.append(best_fitness)
        
        if best_solution is None or objective_function(best_solution) < best_fitness:
            best_solution = population[fitness.index(best_fitness)]
        
        selected_population = select(population, fitness)
        new_population = []
        
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % len(selected_population)]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population[:population_size]
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness_over_time

population_size = 50
generations = 50
mutation_rate = 0.1

best_solution, fitness_over_time = genetic_algorithm(population_size, generations, mutation_rate)
best_coverage = objective_function(best_solution)
coverage_details = get_coverage_details(best_solution)

print(f"Best solution: {best_solution}")
print(f"Maximum Coverage: {best_coverage}")
print("Coverage Details: ")
for facility, demands in coverage_details.items():
    print(f"Facility {facility + 1} covered:{demands}")

plt.plot(range(1, generations + 1), fitness_over_time, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization for UAV Facility Location')
plt.grid(True)
plt.show()
