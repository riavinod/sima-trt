import pandas as pd
import numpy as np
import sys


df = pd.read_csv(sys.argv[1], delimiter=',')

def clean_col(column, col_name):
    lst = []
    #print(column)
    for i in range(1, (column.shape[0])):
        if column[i]==col_name:
            #print('got the col name here')
            continue
        elem = column[i].split()[0]
        #print(elem)
        #print(type(elem))
        #print(int(elem))
        #print(type(int(elem)))
        x = float(elem)
        #print(x, type(x))
        lst.append(x)
    return lst

def get_col_max(df):
    for i in range(df.columns.shape[0]):
        print(df.columns[i], df[df.columns[i]].max())

def get_col_mean(df):
    for i in range(df.columns.shape[0]):
        print(df.columns[i], df[df.columns[i]].mean())

print(df.shape)
#for r in range(df.shape[0]):
    #print(list(df.iloc[r]))
    #print(list(df.columns))
    #if list(df.iloc[r])==list(df.columns):
    #['utilization.gpu [%]', ' utilization.memory [%]', ' memory.total [MiB]', ' memory.free [MiB]', ' memory.used [MiB]']==df.iloc[r].all:
        #print('dropping')
        #print(r)
        #df = df.drop([r], axis=0)
print(df.shape)
col_names = df.columns
dict = {}
for col_name in col_names:
    #print(col_name)
    #print(clean_col(df[col_name]))
    #df[col_name] = clean_col(df[col_name], col_name)
    dict[col_name] = clean_col(df[col_name], col_name)
    print(col_name," max ",  max(dict[col_name]))
    print(col_name, " mean ",sum(dict[col_name])/len(dict[col_name]))        
#print(dict)

#print("Maximum values by metric:")
#get_col_max(df)
#print('\n')
#print("Mean values by metric:")
#get_col_mean(df)
