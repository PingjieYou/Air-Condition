# -*- coding: utf-8 -*-
# @File     : test.py
# @Time     : 2023/5/12 下午4:17
# @Author   : YouPingJie
# @Function :
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义适应度函数
def fitness_function(individual):
    # 将个体转化为超参数
    C = individual[0]
    gamma = individual[1]
    # 创建SVM模型
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    # 训练模型
    svm.fit(X_train, y_train)
    # 计算模型的准确率
    accuracy = svm.score(X_test, y_test)
    return accuracy

# 定义遗传算法参数
population_size = 50
num_generations = 100
mutation_rate = 0.1
num_parents = 10

# 初始化种群
population = []
for i in range(population_size):
    individual = [np.random.uniform(0.1, 10), np.random.uniform(0.1, 1)]
    population.append(individual)

# 遗传算法优化
for generation in range(num_generations):
    # 计算适应度
    fitness_scores = [fitness_function(individual) for individual in population]
    # 选择父代
    parents = [population[i] for i in np.argsort(fitness_scores)[-num_parents:]]
    # 交叉和变异产生下一代
    next_population = parents
    while len(next_population) < population_size:
        parent1 = parents[np.random.randint(0, num_parents)]
        parent2 = parents[np.random.randint(0, num_parents)]
        child = [parent1[0], parent2[1]]  # 交叉操作
        if np.random.rand() < mutation_rate:  # 变异操作
            child[0] = np.random.uniform(0.1, 10)
        if np.random.rand() < mutation_rate:
            child[1] = np.random.uniform(0.1, 1)
        next_population.append(child)
    population = next_population

# 计算最终适应度
fitness_scores = [fitness_function(individual) for individual in population]

# 选择最佳个体
best_individual = population[np.argmax(fitness_scores)]

# 使用最佳个体创建SVM模型
best_C = best_individual[0]
best_gamma = best_individual[1]
best_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
best_svm.fit(X_train, y_train)

# 在测试集上测试模型
accuracy = best_svm.score(X_test, y_test)
print("最佳模型的超参数C和gamma分别为：", best_C, best_gamma)
print("最佳模型的测试准确率为：", accuracy)