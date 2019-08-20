# 个人学习记录以及实践
# 记学习 ng 的机器学习课程记录
# dataset from https://www.kaggle.com/harlfoxem/housesalesprediction/downloads/housesalesprediction.zip/1

# 这个dataset 是房价相关的

import os
import pandas as pd

df = pd.read_csv("input/kc_house_data.csv")
print(df.head())
