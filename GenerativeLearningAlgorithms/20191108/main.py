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
df = df[['class', 'cap-shape', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing']];
key = ['cap_shape_', 'cap_color_', 'bruises_', 'odor_', 'gill-attachment_', 'gill-spacing_']

# type to int 
# cap_shape_mapping = {
#     'b' : 1,
#     'c' : 2,
#     'x' : 3,
#     'f' : 4,
#     'k' : 5,
#     's' : 6
# }
# cap_color_mapping = {
#     'n' : 1,
#     'b' : 2,
#     'c' : 3,
#     'g' : 4,
#     'r' : 5,
#     'p' : 6,
#     'u' : 7,
#     'e' : 8,
#     'w' : 9,
#     'y' : 10 
# }
# class_mapping = {
#     'e' : 0,
#     'p' : 1
# }
# 重新赋值 将字符变量变成数值变量
# df['class'] = df['class'].map(class_mapping)
# df['cap-shape'] = df['cap-shape'].map(cap_shape_mapping)
# df['cap-color'] = df['cap-color'].map(cap_color_mapping)
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

#概率list
p = {}
total_p = {}
train_df = train_df.tolist()
validation_df = validation_df.tolist()
for i in range(len(train_df)):
    for j in range(len(key)):
        key1 = key[j] + str(train_df[i][j+1])
        if key1 in p:
            if train_df[i][0] in p[key1]:
                p[key1][train_df[i][0]] = p[key1][train_df[i][0]] + 1
            else :
                p[key1][train_df[i][0]] = 1
        else :
            p[key1] = {train_df[i][0] : 1}
    if train_df[i][0] in total_p:
        total_p[train_df[i][0]] = total_p[train_df[i][0]] + 1
    else :
        total_p[train_df[i][0]] = 1
#加上拉普拉斯平滑
for i in p:
    for j in p[i]:
        p[i][j] = (p[i][j] + 1) / (total_p[j] + len(train_df))

for i in total_p:
    total_p[i] = total_p[i] / m

print('train set p:', p)
print('total set p:', total_p)

#开始判断 判断验证集是否正确
#p(e|x) =  p(e) * p(xi|e) / p(xi)
#p(p) = 
right_num = 0
error_num = 0
validation_num = len(validation_df)
print('start validat.....\n')
for i in range(validation_num):
    data = validation_df[i]
    #各个后验概率的字典
    validation_p = {}
    #p(x)的多元概率和
    total_validation_p = 0
    for j in total_p:
        #分类器可能不止2种
        _p_x_i = 1
        #计算特征p(xi|y)的概率
        for z in range(len(key)):
            key1 =  key[z] + str(data[z + 1])
            if key1 not in p:
                 #加上拉普拉斯平滑
                _p_x_i = _p_x_i * (1 / (len(train_df) + len(key)))
            else :
                if j not in p[key1]:
                    #加上拉普拉斯平滑
                    _p_x_i = _p_x_i * (1 / (len(train_df) + len(key)))
                else :
                    _p_x_i = _p_x_i * p[key1][j]
        _validation_p = total_p[j] * _p_x_i
        validation_p[j] = _validation_p
        total_validation_p = total_validation_p + _validation_p
    _max = 0
    _validation_result = ''
    #这一步应该可以省略
    for j in total_p:
        validation_p[j] = validation_p[j] / total_validation_p
    for j in validation_p:
        if validation_p[j] > _max:
            _max = validation_p[j]
            _validation_result = j
    print('validat data :', data)
    print('out of test result :', _validation_result)
    if data[0] == _validation_result:
        print('result correct')
        right_num = right_num + 1
    else :
        print('result fail')
        print('validation_p is :', validation_p)
        error_num = error_num + 1
print('run over...')
print('validation num:', validation_num)
print('validation right num', right_num)
print("validation right rate %s" %str(right_num / validation_num * 100))
print('validation error num', error_num)
print("validation error rate %s" %str(error_num / validation_num * 100))
