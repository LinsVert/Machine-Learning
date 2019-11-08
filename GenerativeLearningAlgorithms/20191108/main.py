import pandas as pd
# import matplotlib.pyplot as plt  # 绘图库

import numpy.matlib 
import numpy as np

import os

# data from https://www.kaggle.com/uciml/mushroom-classification
df = pd.read_csv("./../../LogisticRegression/20191018/input/mushrooms.csv")

print('size:', df.shape)

#拆解训练集 与 验证集
#训练集
train_df = []
#验证集
validation_df = []

m = len(df)

print('total:', m)

#使用自组法 获取训练集和验证集 (行数)
validation_list = np.random.choice(m, m)
total_list = np.arange(m)
train_list = np.setdiff1d(total_list, validation_list)
print(train_list.shape)

#格式化数据 先用2个变量进行分类

#class : edible=e, poisonous=p
# cap-shape : bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
df = df[['class', 'cap-shape', 'cap-color']];

# type to int 
cap_shape_mapping = {
    'b' : 1,
    'c' : 2,
    'x' : 3,
    'f' : 4,
    'k' : 5,
    's' : 6
}
cap_color_mapping = {
    'n' : 1,
    'b' : 2,
    'c' : 3,
    'g' : 4,
    'r' : 5,
    'p' : 6,
    'u' : 7,
    'e' : 8,
    'w' : 9,
    'y' : 10 
}
class_mapping = {
    'e' : 0,
    'p' : 1
}
# 重新赋值 将字符变量变成数值变量
df['class'] = df['class'].map(class_mapping)
df['cap-shape'] = df['cap-shape'].map(cap_shape_mapping)
df['cap-color'] = df['cap-color'].map(cap_color_mapping)
df = np.array(df)
#获取训练集
for i in range(len(train_list)):
    train_df.append(df[train_list[i]])   
train_df = np.array(train_df)

for i in range(len(validation_list)):
    validation_df.append(df[validation_list[i]])   
validation_df = np.array(validation_df)
print(validation_df.shape)

#计算各个参数的概率