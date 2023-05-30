# -*- coding: utf-8 -*-
# @File     : svm_gamma_plot.py
# @Time     : 2023/5/30 下午1:33
# @Author   : YouPingJie
# @Function :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 生成随机数据集
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)

# 绘制原始数据集
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Original Data Distribution')

# 定义不同的gamma值
gammas = [0.1, 1, 10]

# 绘制不同gamma值下的决策边界
for gamma in gammas:
    # 创建SVM模型
    clf = SVC(kernel='rbf', gamma=gamma)
    clf.fit(X, y)
    # 绘制决策边界和支持向量
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'SVM with gamma={gamma}')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'SVM Decision Boundary with gamma={gamma}')

# 显示图像
plt.show()