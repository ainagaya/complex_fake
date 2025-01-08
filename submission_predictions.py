import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

    
dfa_train = pd.read_csv('./predictions_1.csv')
df_text = pd.read_csv('./predictions_2.csv')

df = pd.concat([dfa_train,df_text],ignore_index = True)
#df = df.drop(columns=['ColumnName'],axis=0)
df.index.name = 'Id'
df.to_csv('submission_file.csv',sep=',',header=True)