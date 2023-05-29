# -*- coding: utf-8 -*-
# @File     : ga_svm.py
# @Time     : 2023/5/12 下午8:20
# @Author   : YouPingJie
# @Function : 遗传算法优化SVM的超参数
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import random
import numpy as np
from sklearn.svm import SVC


class Individual():
    def __init__(self, attr_num):
        self.attr_num = attr_num
        self.parameter = [np.random.uniform(0.1, 10), np.random.uniform(0.1, 1)]
        self.fitness = 0
        self.svm = None

    def get_info(self):
        print("C and gamma: ", self.parameter, "fitness:", self.fitness)

    def update_fitness(self, train_data, test_data):
        C = self.parameter[0]
        gamma = self.parameter[1]

        self.svm = SVC(C=C, gamma=gamma, kernel='rbf')
        self.svm.fit(train_data[0], train_data[1])
        return self.svm.score(test_data[0], test_data[1])


class GeneticAlgorithm():
    def __init__(self):
        self.pop_size = 50
        self.max_gen = 20
        self.num_parent = 10
        self.mutation_rate = 0.1

    def random_select(self, pop):
        """
        随机选择

        :param pop:
        :return:
        """
        fitness_sum = sum([individual.fitness for individual in pop])
        pro_pop = [individual.fitness / fitness_sum for individual in pop]

        idx = []

        while len(idx) < self.num_parent:
            r = random.uniform(0, 1)  # 模拟赌盘生产随机数
            sm = 0

            for i in range(len(pro_pop)):
                sm += pro_pop[i]
                if sm >= r:
                    if i in idx:
                        break
                    idx.append(i)
                    break
        return [pop[i] for i in idx]

    def crossover(self, parent_1, parent_2):
        """
        交叉

        :param parent_1:
        :param parent_2:
        :return:
        """
        child_parameter = [parent_1.parameter[0], parent_2.parameter[1]]

        return child_parameter

    def mutate(self, individual):
        """
        变异

        :param individual:
        :return:
        """
        if np.random.rand() < self.mutation_rate:
            individual.parameter[0] = np.random.uniform(0.1, 10)
        if np.random.rand() < self.mutation_rate:
            individual.parameter[1] = np.random.uniform(0.1, 1)
        return individual


def main():
    # 加载数据集
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 算法
    ga = GeneticAlgorithm()

    # 初始化种群
    pop = []
    for i in range(ga.pop_size):
        individual = Individual(2)
        pop.append(individual)

    # 遗传算法优化
    for generation in range(ga.max_gen):
        # 计算适应度
        for individual in pop:
            individual.fitness = individual.update_fitness([X_train, y_train], [X_test, y_test])
        # 选择
        # parents = [pop[i] for i in np.argsort([individual.fitness for individual in pop])[-ga.num_parent:]]
        parents = ga.random_select(pop)

        # 产生一下代
        next_pop = parents
        while len(next_pop) < ga.pop_size:
            parent1 = parents[np.random.randint(0, ga.num_parent)]
            parent2 = parents[np.random.randint(0, ga.num_parent)]
            child = Individual(2)
            # 交叉
            child.parameter = ga.crossover(parent1, parent2)
            # 变异
            child = ga.mutate(child)
            next_pop.append(child)
        pop = next_pop
        print("Generation: ", generation, "Best fitness: ", max([individual.fitness for individual in pop]))
        print("Best individual: ",end='')
        pop[np.argmax([individual.fitness for individual in pop])].get_info()
        print("--------------------------------------------------")

    # 计算最终种群的适应度
    for individual in pop:
        individual.fitness = individual.update_fitness([X_train, y_train], [X_test, y_test])

    # 选择最佳的个体
    best_individual = pop[np.argmax([individual.fitness for individual in pop])]
    print("Best individual: ", end='')
    best_individual.get_info()

    # 利用最佳个体构建SVM模型
    best_C = best_individual.parameter[0]
    best_gamma = best_individual.parameter[1]
    best_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
    best_svm.fit(X_train, y_train)
    print("Accuracy on test set: ", best_svm.score(X_test, y_test))

if __name__ == '__main__':
    main()
