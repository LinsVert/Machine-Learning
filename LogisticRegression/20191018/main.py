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
df = df[['class', 'cap-shape', 'cap-color', 'bruises', 'odor', 'cap-surface']];
thea_num = 6
# type to int 
cap_shape_mapping = {
    'b' : 0.5,
    'c' : 1.0,
    'x' : 1.5,
    'f' : 2.0,
    'k' : 2.5,
    's' : 3.0
}
cap_color_mapping = {
    'n' : 0.5,
    'b' : 1.0,
    'c' : 1.5,
    'g' : 2.0,
    'r' : 2.5,
    'p' : 3.0,
    'u' : 3.5,
    'e' : 4.0,
    'w' : 4.5,
    'y' : 5.0 
}
bruises_mapping = {
    'f' : 1.0,
    't' : 2.0
}
odor_mapping = {
    'a' : 0.5,
    'l' : 1.0,
    'c' : 1.5,
    'y' : 2.0,
    'f' : 2.5,
    'm' : 3.0,
    'n' : 3.5,
    'p' : 4.0,
    's' : 4.5,
}
cap_surface_mapping = {
    'f' : 0.5,
    'g' : 1.0,
    'y' : 1.5,
    's' : 2.0
}
class_mapping = {
    'e' : 0,
    'p' : 1
}
# 重新赋值 将字符变量变成数值变量
df['class'] = df['class'].map(class_mapping)
df['cap-shape'] = df['cap-shape'].map(cap_shape_mapping)
df['bruises'] = df['bruises'].map(bruises_mapping)
df['cap-color'] = df['cap-color'].map(cap_color_mapping)
df['odor'] = df['odor'].map(odor_mapping)
df['cap-surface'] = df['cap-surface'].map(cap_surface_mapping)
x = df[['cap-shape', 'cap-color', 'bruises', 'odor', 'cap-surface']]
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

# x2 = df[['cap-shape']] * df[['cap-shape']]
# x3 = df[['cap-color']] * df[['cap-color']]
y = df[['class']]
m = len(x)
a = 0.01 #学习率
x = np.asmatrix(x)
y = np.asmatrix(y)
thea = np.matlib.zeros((thea_num, 1))
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
        sum = sum * (-1)
        return (sum.tolist())[0][0]
# def h(thea, x):
#         return 0
# diff = y - np.dot(x, thea)
diff = np.dot(x, thea)
diff = g(diff) - y
diff = np.dot(np.transpose(x), diff) * (1./m)
loop_time = 0
diff_arr = []
while not np.all(np.absolute(diff) <= 1e-5) and loop_time < 20:
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
new_theta = {}
for j in range(thea_num):
    key = 'thea_' + str(j)
    new_theta[key] = []
for i in thea_total:
    b = i.tolist()
    for j in range(thea_num):
        key = 'thea_' + str(j)
        new_theta[key].append(b[j][0])
    # c = cost_func(i, x, y, m)
    # print(c)
    # cost.append(c)
#画出代价函数
plt.figure()
for i in range(thea_num):
    strs = "theta " + str(i + 1)
    key = 'thea_' + str(i)
    plt.plot(new_theta[key], cost, label=strs)
plt.xlabel('thea值')
plt.ylabel('代价函数值cost')
#画出预测函数
plt.figure()
z = np.dot(x, thea)
h = g(z)
z = z.tolist()
h = h.tolist()
y = y.tolist()
new_h = []
new_z = []
new_y = []
error_num = 0
right_num = 0
for i in range(m):
    c = 'r'
    if y[i][0] is 0:
        c = 'g'
    if h[i][0] < 0.5 and y[i][0] is 0:
        right_num = right_num + 1
    elif h[i][0] >= 0.5 and y[i][0] is 1:  
        right_num = right_num + 1
    else :
        error_num = error_num + 1
    new_h.append(h[i][0])
    new_z.append(z[i][0])
    new_y.append(c)
print('right num:', right_num)
print('right rate :%s' %str(right_num / m * 100))
print('error rate :', error_num)
print('error num :%s' %str(error_num / m * 100))
plt.scatter(new_z, new_h, c = new_y)
plt.show()