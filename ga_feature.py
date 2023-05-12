# -*- coding: utf-8 -*-
# @File     : ga_feature.py
# @Time     : 2023/5/12 上午10:04
# @Author   : YouPingJie
# @Function : 遗传算法优化获得最适合SVM的特征

import copy
from functools import cmp_to_key

import utils
import random
import data_process
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


class GeneticAlgorithm():
    def __init__(self):
        self.pop_size = 30
        self.max_gen = 20

    def generate_code(self, attr_num):
        """
        生成二进制编码

        :param attr_num: 属性数量
        :return:
        """
        return ''.join([random.choice(['0', '1']) for _ in range(attr_num)])

    def compare(self, a, b):
        """
        比较函数

        :param a:
        :param b:
        :return:
        """
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0

    def random_select(self, pop):
        """
        随机选择

        :param pop:
        :return:
        """
        fitness_sum = sum([individual.fitness for individual in pop])
        pro_pop = [individual.fitness / fitness_sum for individual in pop]

        idx = []

        while len(idx) < 2:
            r = random.uniform(0, 1)  # 模拟赌盘生产随机数
            sm = 0

            for i in range(len(pro_pop)):
                sm += pro_pop[i]
                if sm >= r:
                    if i in idx:
                        break
                    idx.append(i)
                    break
        return idx

    def crossover(self, parent_1, parent_2, rate=0.8, min_num=3, max_num=10):
        """
        交叉

        :param par_1:
        :param par_2:
        :param rate:
        :param min_num:
        :param max_num:
        :return:
        """
        cross_pos = []
        child_1 = parent_1
        child_2 = parent_2

        child_1.code = list(child_1.code)
        child_2.code = list(child_2.code)

        for x in range(0, len(parent_1.code)):
            if random.random() < rate:
                cross_pos.append(x)

        for x in cross_pos:
            child_1.code[x] = parent_2.code[x]
            child_2.code[x] = parent_1.code[x]

        child_1.code = ''.join(child_1.code)
        child_2.code = ''.join(child_2.code)
        return [child_1, child_2]

    def mutate(self, individual, rate=0.02):
        """
        变异

        :param rate:
        :return:
        """
        individual.code = list(individual.code)
        for x in range(0, len(individual.code)):
            if random.random() < rate:
                if individual.code[x] == '0':
                    individual.code[x] = '1'
                else:
                    individual.code[x] = '0'
        individual.code = ''.join(individual.code)
        return individual


class Individual():
    def __init__(self, attr_num):
        self.attr_num = attr_num
        self.code = ''
        self.fitness = 0
        self.status = []
        self.svm = LinearSVC(C=1, loss='hinge')

    def get_info(self):
        print("code:", self.code, "fitness:", self.fitness, "status:", self.status)

    def update_code(self, ga):
        self.code = ga.generate_code(self.attr_num)

    def update_status(self):
        self.status = []
        for i in range(len(self.code)):
            if self.code[i] == '1':
                self.status.append(i)

    def update_fitness(self, train_data, test_data):
        self.svm.fit(train_data[0], train_data[1])
        self.fitness = self.svm.score(test_data[0], test_data[1])
        print(self.svm.C)

def main():
    df = utils.csv2df('csv/data.csv')
    X, y = data_process.get_cls_data(df)
    X = data_process.pre_processing(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 算法
    ga = GeneticAlgorithm()

    # 初始化种群
    pop = []
    for _ in range(ga.pop_size):
        individual = Individual(len(X_train[0]))
        individual.update_code(ga)
        individual.update_status()
        individual.update_fitness((X_train, y_train), (X_test, y_test))
        pop.append(individual)
        print("已初始化", _ + 1, "个个体")

    # 遗传算法优化svm
    best_invividual = []
    best_1 = copy.deepcopy(sorted(pop, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)), reverse=True)[0])
    best_2 = copy.deepcopy(sorted(pop, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)), reverse=True)[1])
    for generation in range(ga.max_gen):
        best_individual_1 = copy.deepcopy(best_1)
        best_individual_2 = copy.deepcopy(best_2)
        individual_pop = []
        individual_pop.append(best_individual_1)
        individual_pop.append(best_individual_2)
        while len(individual_pop) < ga.pop_size:
            # 随机选择父带
            [parent1_idx, parent2_idx] = ga.random_select(pop)
            parent1, parent2 = pop[parent1_idx], pop[parent2_idx]
            # 交叉
            [child_1, child_2] = ga.crossover(parent1, parent2)
            # 变异
            child_1 = ga.mutate(child_1, 0.5)
            child_2 = ga.mutate(child_2, 0.5)
            # 计算子代适应度
            child_1.update_fitness((X_train, y_train), (X_test, y_test))
            child_2.update_fitness((X_train, y_train), (X_test, y_test))
            # 加入种群
            individual_pop.append(child_1)
            individual_pop.append(child_2)
            individual_pop = list(filter(None.__ne__, individual_pop))
        # 更新最优个体
        best_individual_list = [sorted(individual_pop, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)))[0],
                                sorted(individual_pop, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)))[1],
                                best_individual_1, best_individual_2]
        best_individual_1 = copy.deepcopy(sorted(best_individual_list, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)))[0])
        best_individual_2 = copy.deepcopy(sorted(best_individual_list, key=cmp_to_key(lambda x, y: ga.compare(x.fitness, y.fitness)))[1])
        pop = individual_pop
        for individual in individual_pop:
            individual.get_info()


if __name__ == '__main__':
    main()
