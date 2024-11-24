import numpy as np
import random
import matplotlib.pyplot as plt

# 固定参数定义
I = 10  # 需求点数量，手动指定
J = 5  # 医院数量
M = 100 # 总预算
C = 10  # 单个平台建设费用
T = 9   # 最大响应时间（分钟）
V = 0.8 # 无人机速度

# 固定的距离矩阵和需求量
# 需求点到设施点的距离矩阵 (I x J)
dij = [
    [10, 15, 12, 9, 8],
    [5, 9, 7, 8, 6],
    [8, 12, 11, 13, 9],
    [6, 10, 8, 7, 15],
    [9, 11, 14, 8, 7],
    [12, 13, 10, 9, 6],
    [11, 14, 12, 15, 10],
    [7, 5, 9, 8, 13],
    [6, 9, 7, 11, 12],
    [10, 8, 6, 9, 11]
]

# 需求量 (I)
wi = [3, 4, 5, 2, 6, 3, 4, 5, 2, 7]

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

# 轮盘赌
def select(population, fitness):  # 获取适应度列表中的最小值
    min_fitness = min(fitness) 
    if min_fitness < 0: # 如果存在负适应度值，将所有适应度值平移为非负值
        fitness = [f - min_fitness for f in fitness] # 将每个适应度值加上 min_fitness 的绝对值
    total_fitness = sum(fitness) # 计算总适应度
    probabilities = [f / total_fitness for f in fitness] # 计算每个个体的选择概率，适应度值越大概率越高
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities) # 随机选择
    return [population[i] for i in selected_indices] # 将选中的提取出来得到新种群

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
print("最佳设施组合:", best_solution)
print("最大覆盖的需求总量:", best_solution_value)
print("需求点数量:", I)
print("每个需求点的需求量:", wi)
print("总需求量:", sum(wi))

# 绘制适应度变化图
plt.plot(fitness_over_time)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization for Facility Location')
plt.show()
