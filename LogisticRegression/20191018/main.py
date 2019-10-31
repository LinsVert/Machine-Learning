import pandas as pd
import matplotlib.pyplot as plt  # 绘图库

import numpy.matlib 
import numpy as np

import os

# data from https://www.kaggle.com/uciml/mushroom-classification
df = pd.read_csv("./input/mushrooms.csv")

class_type = df[['class']]
# print(class_type)
#
# 简单的二元分类 多元分类todo 
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
# print(df.head())
#获取类别 进行画图
# df_p = df[df['class'].isin([1])].drop_duplicates()
# x_p_s = df_p[['cap-shape']]
# x_p_c = df_p[['cap-color']]
# df_e = df[df['class'].isin([0])].drop_duplicates()
# x_e_s = df_e[['cap-shape']]
# x_e_c = df_e[['cap-color']]

# df = df.drop_duplicates()
# print(df)

#绘制 x1 = shape x2 = color 二元分类图
# plt.scatter(x_e_s, x_e_c, c= 'green', alpha = 0.5) #可使用的蘑菇位置
# plt.scatter(x_p_s, x_p_c, c= 'blue', alpha = 0.1) #有毒的蘑菇位置

# plt.show()
# theta = theta - a * (y - 1 / (1 + e^(theta_T X))) * x(i)

x = df[['cap-shape', 'cap-color']]
# x2 = df[['cap-shape']] * df[['cap-shape']]
# x3 = df[['cap-color']] * df[['cap-color']]
y = df[['class']]
m = len(x)
a = 0.01 #学习率
x = np.asmatrix(x)
y = np.asmatrix(y)
thea = np.matlib.zeros((3, 1))
x0 = np.matlib.ones((m, 1))
x = np.concatenate([x0, x], 1)
cost = []
thea_total = []
# print(y[0])
# os._exit(0)
def g(thea_x):
    	return 1./ (1 + np.exp(-thea_x))

def cost_func(thea, x, y, m, diff):
        # -ylog(y)-(1-y)log(1-y)
        h = diff
        sum = np.dot(np.transpose(y), np.log(h)) * (-1)
        sum = sum - np.dot(1 - np.transpose(y),np.log(1 - h))
        sum = 1. / m * sum
        return (sum.tolist())[0][0]
# def h(thea, x):
#         return 0
# diff = y - np.dot(x, thea)
diff = np.dot(x, thea)
diff = g(diff) - y
diff = np.dot(np.transpose(x), diff) * (1./m)
loop_time = 0
diff_arr = []
while not np.all(np.absolute(diff) <= 1e-5) and loop_time < 1000:
        # loop
        thea = thea - a * diff
        diff = np.dot(x, thea)
        diff = g(diff)
        cost.append(cost_func(thea, x, y, m, diff))
        diff = diff - y
        diff = np.dot(np.transpose(x), diff) * (1./m)
        # loop_time = loop_time + 1
       
        thea_total.append(thea)
        # loop_time = loop_time + 1
        print('thea: ', thea)
print('final thea', thea)
new_theta_1 = []
new_theta_2 = []
new_theta_3 = []
for i in thea_total:
    b = i.tolist()
    new_theta_1.append(b[0][0])
    new_theta_2.append(b[1][0])
    new_theta_3.append(b[2][0])
    # c = cost_func(i, x, y, m)
    # print(c)
    # cost.append(c)
plt.plot(new_theta_1, cost)
plt.plot(new_theta_2, cost)
plt.plot(new_theta_3, cost)
plt.xlabel('thea值')
plt.ylabel('代价函数值cost')
plt.show()