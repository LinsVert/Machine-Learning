#最小二乘法
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib 
import numpy as np

df = pd.read_csv("../20190820/input/kc_house_data.csv")
#因是简单的一个线性回归 参数先使用2个 price 和 sqft_living
#需要以下矩阵 价格结果矩阵 面积矩阵
y = df[['price']]
x = df[['sqft_living']]
m = len(x)
#将df 转换成矩阵
y = np.asmatrix(y) / 1000
x = np.asmatrix(x) / 1000
x0 = np.matlib.ones((m, 1))
#合并矩阵 2 x m
x = np.concatenate([x0, x], 1)

x1 = 1. / np.dot(np.transpose(x), x)
x2 = np.dot(x1, np.transpose(x))
x3 = np.dot(x2, y)
print(x3)