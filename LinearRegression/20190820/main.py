# 个人学习记录以及实践
# 记学习 ng 的机器学习课程记录
# dataset from https://www.kaggle.com/harlfoxem/housesalesprediction/downloads/housesalesprediction.zip/1

# 这个dataset 是房价相关的

import os
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

plt.figure()

plt.scatter(x, y, s=5, c='red', label = 'Predicted Regression Line')

plt.xlabel('Living Space (sqft)')
plt.ylabel('Price ($)')
plt.show()
