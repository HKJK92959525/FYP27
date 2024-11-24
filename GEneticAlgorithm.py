import numpy as np
import random
import matplotlib.pyplot as plt

# 固定参数定义
I = 15  # 需求点数量，手动指定
J = 6  # 医院数量
M = 100 # 总预算
C = 50  # 单个平台建设费用
T = 9   # 最大响应时间（分钟）
V = 1 # 无人机速度

# 固定的距离矩阵和需求量
# 需求点到设施点的距离矩阵 (I x J)
dij = [
  [15.13, 31.14, 35.51, 11.4, 24.17, 41.76],
  [16.55, 20.12, 24, 33.54, 17, 12.21],
  [10.63, 17.03, 26.25, 25.5, 16, 18.97],
  [20.02, 36.77, 38.48, 14.14, 27.89, 47.43],
  [23.85, 19.65, 46.4, 8.6, 34.41, 38.47],
  [27.29, 2.83, 45.35, 26.08, 35.13, 24.7],
  [32.57, 7.21, 48.26, 33.29, 39.12, 22.67],
  [17.46, 43.17, 19, 33.11, 14.21, 42.45],
  [29.15, 9.85, 49.4, 21.93, 38.33, 31.76],
  [24.84, 11.66, 36.01, 34.53, 28.46, 10.3],
  [35.81, 23.35, 40.5, 47.68, 36.4, 8.06],
  [25.46, 49.24, 33.02, 31, 26.93, 53.6],
  [24.41, 41.44, 5.83, 46.62, 14.32, 28.44],
  [27.17, 35, 18.44, 47.76, 20.62, 17.12],
  [28.07, 8.54, 48.1, 21.93, 37.11, 30.41]
]

# 需求量 (I)
wi = [4, 4, 2, 1, 9, 2, 3, 5, 1, 5, 3, 2, 6, 7, 5]

# 需求点和医院的位置坐标（从用户提供的数据中读取）
demand_points = [(45, 21), (15, 26), (23, 27), (50, 18), (43, 39), (26, 46), (20, 50), (33, 48), (14, 38), (1, 40), (48, 1), (10, 5), (3, 16), (10, 8), (32, 47)]
facility_points = [(30, 19), (24, 44), (15, 2), (48, 32), (23, 11), (5, 33)]

# 可视化需求点和医院的位置
def plot_points():
    plt.figure(figsize=(8, 6))
    
    # 绘制需求点
    for i, (x, y) in enumerate(demand_points):
        plt.scatter(x, y, c='blue', marker='o')
        plt.text(x + 0.1, y + 0.1, f'Demand {i + 1}', fontsize=9)
    
    # 绘制医院位置
    for j, (x, y) in enumerate(facility_points):
        plt.scatter(x, y, c='red', marker='^')
        plt.text(x + 0.1, y + 0.1, f'Facility {j + 1}', fontsize=9)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locations of Demand Points and Facilities')
    plt.grid(True)
    plt.show()

# 目标函数：计算总覆盖需求量
def objective_function(x):
    total_demand_covered = 0 # 初始覆盖量为 0
    for i in range(I):
        is_covered = any(x[j] == 1 and dij[i][j] / V <= T for j in range(J)) # 检查是否被覆盖
        if is_covered:
            total_demand_covered += wi[i] # 若被覆盖则增加到总覆盖里
    return total_demand_covered # 返回总覆盖

# 初始化种群
def initialize_population(size, bounds): # 定义一个初始化种群的函数，输入参数有种群规模 (size) 和变量范围 (bounds)
    population = [] # 空列表用于存储生成的个体
    for _ in range(size):
        individual = [random.choice([0, 1]) for _ in range(J)]
        # 创建一个个体，该个体由 J 个随机选择的 0 或 1 组成。
        if sum(individual) * C <= M:  # 检查是否满足预算约束
            population.append(individual)  # 将满足预算的个体加入到 population 里
    return population

# 适应度评估
def evaluate_population(population):
    # 对种群中的每个个体 individual 调用 objective_function，得到其适应度值，
    # 并将这些适应度值组成一个列表返回
    return [objective_function(individual) for individual in population]

# 轮盘赌选择函数
def select(population, fitness):
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]  # 将所有适应度值平移为非负值
    total_fitness = sum(fitness)
    
    # 如果总适应度为 0，避免除以零的错误
    if total_fitness == 0:
        # 直接返回种群，或者随机选择一些个体
        return random.choices(population, k=len(population))
    
    # 计算每个个体的选择概率
    probabilities = [f / total_fitness for f in fitness]
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in selected_indices]


# 交叉
def crossover(dad, mom):
    child1 = [(p1 + p2) // 2 for p1, p2 in zip(dad, mom)] # 子代1的基因为父代和母代基因的均值
    child2 = [(p1 + p2) // 2 for p1, p2 in zip(dad, mom)] # 子代2的基因为父代和母代基因的均值
    return child1, child2

# 变异
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        mutate_point = random.randint(0, J - 1) # 随机选择一个基因位置进行变异
        individual[mutate_point] = 1 - individual[mutate_point]  # 在0和1之间切换
    return individual

# 遗传算法主函数
def genetic_algorithm(bounds, population_size, generations, mutation_rate):
    population = initialize_population(population_size, bounds)
    best_fitness_over_time = []
    best_solution = None
    best_solution_value = -float('inf')

    for generation in range(generations):
        fitness = evaluate_population(population) # 评估种群适应度
        best_fitness = max(fitness)
        best_fitness_over_time.append(best_fitness) # 每一代最佳适应度
        
        # 更新最佳解
        if best_fitness > best_solution_value:
            best_solution_value = best_fitness
            best_solution = population[fitness.index(best_fitness)]
        
        # 打印每代的最佳适应度
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        selected_population = select(population, fitness)
        new_population = [] # 用于存储子代用的新的列表。。。
        
        # 生成新种群 交叉操作
        for i in range(0, len(selected_population), 2):
            dad, mom = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(dad, mom)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population

    # 返回最佳解及其适应度随迭代时间变化
    return best_solution, best_fitness_over_time

# 参数设置
bounds = [0, 1]  
population_size = 20
generations = 50
mutation_rate = 0.1

# 执行遗传算法
best_solution, fitness_over_time = genetic_algorithm(bounds, population_size, generations, mutation_rate)

# 输出最佳解
best_solution_value = objective_function(best_solution)
print("Best Solution:", best_solution)
print("Maximum Coverage:", best_solution_value)
print("Number of Demand Points:", I)
print("Demand Value of Demand Points:", wi)
print("Total Demand Value:", sum(wi))

# 绘制适应度变化图
plt.plot(fitness_over_time)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization for Facility Location')
plt.show()

# 绘制需求点和设施点的位置
plot_points()
