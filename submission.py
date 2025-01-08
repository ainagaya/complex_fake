import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random as rd

list = ["real", "fake"]

# Fixing the seed
rd.seed(11012000)

# Create a CSV file with 256 rows
with open("data/train_A_derma.csv", "w") as file:
    # write header
    file.write("ColumnName" + "\n")
    for row in range(256):
        file.write(rd.choice(list) + '\n')
        
with open("data/test.csv", "w") as file:
    file.write("ColumnName" + "\n")
    for row in range(1050):
        file.write(rd.choice(list) + '\n')
    
dfa_train = pd.read_csv('./data/train_A_derma.csv')
df_text = pd.read_csv('./data/test.csv')

df = pd.concat([dfa_train,df_text],ignore_index = True)
#df = df.drop(columns=['ColumnName'],axis=0)
df.index.name = 'Id'
df.to_csv('submission_file.csv',sep=',',header=True)