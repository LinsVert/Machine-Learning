import os
import pandas as pd

# print(os.listdir("input"))

os.chdir('input')

df = pd.read_csv('2004-2019.tsv', sep='\t',parse_dates=[1,2])

print(df.shape)