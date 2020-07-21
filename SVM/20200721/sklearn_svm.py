from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def hyperplanes(clf, X, Y, step=0.1):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 获取随机值
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))

    plt.title('hyperplanes')
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.xticks(())
    plt.yticks(())

    # 重新预测
    # ravel 转一维 c_ 拼接 成 矩阵
    z = clf.predict(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)
    plt.contourf(x1, x2, z)
    markers = ['o', 's', '^']
    colors = ['r', 'g', 'c']
    labels = np.unique(Y)
    for label in labels:
        _label = 1
        if label == -1:
            _label = 0
        plt.scatter(X[Y == label][:, 0], X[Y == label][:, 1], c=colors[_label], marker=markers[_label])
    sv = clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
    plt.show()


df = pd.read_csv("./input/testSet.txt", names=["x1", "x2", "y"], sep="\t")
print('shape', df.shape)

X1 = df['x1']
X1 = np.transpose(np.array([X1]))
X2 = df['x2']
X2 = np.transpose(np.array([X2]))
X = np.hstack((X1, X2))
Y = df['y']

clf = svm.SVC()
clf.fit(X, Y)

# 决策平面

# W* * X + b = 0

hyperplanes(clf, X, Y)