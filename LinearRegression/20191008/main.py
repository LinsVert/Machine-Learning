#梯度下降矩阵型式

import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib 
import numpy as np

#导入数据

df = pd.read_csv("../20190820/input/kc_house_data.csv")
#因是简单的一个线性回归 参数先使用2个 price 和 sqft_living
#需要以下矩阵 价格结果矩阵 面积矩阵
y = df[['price']]
x = df[['sqft_living']]
m = len(x)
a = 0.01 #学习率   
#将df 转换成矩阵
y = np.asmatrix(y) / 1000
x = np.asmatrix(x) / 1000
#因x0 默认为1 需要一个 1xm的 x0矩阵
x0 = np.matlib.ones((m, 1))
#合并矩阵 2 x m
x = np.concatenate([x0, x], 1)
#定义thea 目前设置为2个变量 默认为0
thea = np.matlib.zeros((2, 1))
diff = np.dot(x, thea) - y
diff = np.dot(np.transpose(x), diff) * (1./m)
while not np.all(np.absolute(diff) <= 1e-5):
	#loop
	thea = thea - a * diff
	diff = np.dot(x, thea) - y
	diff = np.dot(np.transpose(x), diff) * (1./m)
	print('thea: ', thea)
print('final thea', thea)
# thea - a * (thea * x - y) * xT / m
# new_thea = thea - a * (thea * x - y) * x.T / m