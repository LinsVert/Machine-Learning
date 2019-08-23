# 个人学习记录以及实践
# 记学习 ng 的机器学习课程记录
# dataset from https://www.kaggle.com/harlfoxem/housesalesprediction/downloads/housesalesprediction.zip/1

# 这个dataset 是房价相关的

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *  

df = pd.read_csv("input/kc_house_data.csv", )

# print(df[:1])

#因是简单的一个线性回归 参数先使用2个 price 和 sqft_living
df = df[['price', 'sqft_living']]


#1.绘制价格和房间规格大小的点图

alt = array(df)

x = []
y = []
for a in alt:
    x.append(a[1])
    y.append(a[0])
x, y = array(x),array(y)

# plt.figure()

# plt.scatter(x, y, s=5, c='red', label = 'Predicted Regression Line')

# plt.xlabel('Living Space (sqft)')
# plt.ylabel('Price ($)')
# plt.show()

# 定义预测函数 h = z + z1 * x 预测为线性
# 定义学习率 a
a = 0.01
# 定义初始值
z = 0
z1 = 50
# 数量长度
x = x / 1000
y = y / 1000

m = len(x)
# 定义极限
min_limit = 1e-3

def getH(z, z1, x, y):
    return z + z1 * x - y

# 获取m
def getZ(z, z1, x, y, m):
    sum = 0 
    for i in range(m):
        h = getH(z, z1, x[i], y[i])
        sum = sum + h
    sum = sum * (1./ m)
    # newZ = z - sum
    return sum
def getZ1(z, z1, x, y, m):
    sum = 0 
    for i in range(m):
        h = getH(z, z1, x[i], y[i])
        h = h * x[i]
        sum = sum + h
    sum = sum * (1./ m)
    # newZ1 = z1 - sum
    return sum
def getAllZ(z, z1, x, y, m, a, min_limit):
    newZ = getZ(z, z1, x, y, m)
    newZ1 = getZ1(z, z1, x, y, m)
    while abs(newZ) >= min_limit and abs(newZ1) >= min_limit:
        print('z z1', z, z1)
        print('new z new z1', newZ, newZ1)
        z = z - a * newZ
        z1 = z1 - a * newZ1
        newZ = getZ(z, z1, x, y, m)
        newZ1 = getZ1(z, z1, x, y, m)
    print('new z', newZ)
    print('new z1', newZ1)
    print('z z1', z, z1)
    return [z, z1]

result = getAllZ(z, z1, x, y, m, a, min_limit)

#10条数据 z = -13.80519903101423 z1 = 225.65934680567906
#全部数据 z =  -43.56346904629207 280.61642450710656
def testY(x, result):
    return (result[0] + result[1] * x);
x = x * 1000
y = y * 1000
text_y = []
for i in x:
    # print(i)
    text_y.append(testY(i, result))

plt.figure()
plt.scatter(x, y, s=5, c='red', label='Data')
plt.plot(x, text_y, c='darkgreen', label='Predicted Regression Line')
plt.xlabel('Living Space (sqft)')
plt.ylabel('Price ($)')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.show()
