# -*- coding: utf-8 -*-
# @File     : svm_c_plot.py
# @Time     : 2023/5/30 下午1:27
# @Author   : YouPingJie
# @Function :

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 只使用前两个特征
y = iris.target

# 定义不同的C值
C_values = [0.01, 1, 10,100,1000]

# 可视化数据集
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# 构建并绘制不同C值下的SVM模型
for C in C_values:
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    # 绘制决策边界
    plt.plot([X[:, 0].min(), X[:, 0].max()], [(-svc.intercept_[0]-svc.coef_[0][0]*X[:, 0].min())/svc.coef_[0][1], (-svc.intercept_[0]-svc.coef_[0][0]*X[:, 0].max())/svc.coef_[0][1]], label='C = {}'.format(C))

plt.legend()
plt.title('SVM with different C values')
plt.show()

