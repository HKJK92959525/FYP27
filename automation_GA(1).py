import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取 CSV 文件并转换为距离矩阵
def load_distance_matrix(csv_file):
    df = pd.read_csv(csv_file, index_col=0, encoding = 'ISO-8859-1')  # 读取文件，并将第一列作为索引
    distance_matrix = df.values  # 转换为 numpy 矩阵
    return distance_matrix

# CSV 文件路径（修改为你的文件路径）
csv_file = "HK_nursing_house_distance_matrix.csv"
dij = load_distance_matrix(csv_file)  # 自动加载 dij 矩阵

# 问题参数
I, J = dij.shape  # 需求点数量 = 候选设施数量 = 矩阵的大小
M = 1000      # 总预算
C = 200       # 每个设施的建设费用
T = 9        # 最大响应时间（分钟）
V = 0.6      # UAV 速度（这里单位与 T 配合后，max_distance = V * T）

# 最大服务距离（单位与 dij 相同）
max_distance = V * T

# 生成随机需求量
wi = [random.randint(1, 10) for _ in range(I)]

# 目标函数：计算一个解下被覆盖需求总量
def objective_function(solution):
    total_covered_demand = 0
    for i in range(I):
        is_covered = any(solution[j] == 1 and dij[i][j] <= max_distance for j in range(J))
        if is_covered:
            total_covered_demand += wi[i]
    return total_covered_demand

# 获取覆盖详情
def get_coverage_details(solution):
    coverage_details = {j: [] for j in range(J) if solution[j] == 1}
    for i in range(I):
        for j in range(J):
            if solution[j] == 1 and dij[i][j] <= max_distance:
                coverage_details[j].append(i + 1)  # 需求点编号从1开始
    return coverage_details

# 初始化种群，确保预算约束：选中设施数量 * C <= M
def initialize_population(size):
    population = []
    for _ in range(size):
        facilities = [random.choice([0, 1]) for _ in range(J)]
        while sum(facilities) * C > M:
            facilities[random.randint(0, J - 1)] = 0
        population.append(facilities)
    return population

# 评估种群适应度
def evaluate_population(population):
    return [objective_function(individual) for individual in population]

# 锦标赛选择
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# 单点交叉
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 变异
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    while sum(individual) * C > M:
        individual[random.randint(0, len(individual) - 1)] = 0
    return individual

# 遗传算法
def genetic_algorithm(population_size, generations, mutation_rate):
    population = initialize_population(population_size)
    best_fitness_over_time = []
    best_solution = None

    for generation in range(generations):
        fitness = evaluate_population(population)
        best_fitness = max(fitness)
        best_index = fitness.index(best_fitness)
        best_individual = population[best_index].copy()
        best_fitness_over_time.append(best_fitness)

        if best_solution is None or objective_function(best_solution) < best_fitness:
            best_solution = best_individual.copy()

        selected_population = tournament_selection(population, fitness, tournament_size=3)
        new_population = []

        # 精英保留
        new_population.append(best_individual.copy())

        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))
            if len(new_population) >= population_size:
                break

        while len(new_population) < population_size:
            new_population.append(random.choice(population))
        population = new_population[:population_size]

        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    return best_solution, best_fitness_over_time

# 运行遗传算法
population_size = 100
generations = 50
mutation_rate = 0.1

best_solution, fitness_over_time = genetic_algorithm(population_size, generations, mutation_rate)
best_coverage = objective_function(best_solution)
coverage_details = get_coverage_details(best_solution)

print(f"Best solution: {best_solution}")
print(f"Maximum Coverage: {best_coverage}")
print("Coverage Details:")
for facility, demands in coverage_details.items():
    print(f"Facility {facility + 1} covered: {demands}")

# 绘制适应度变化
plt.plot(range(1, generations + 1), fitness_over_time, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization for UAV Facility Location')
plt.grid(True)
plt.show()
