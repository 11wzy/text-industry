# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:38:16 2023

@author: a
"""

import os
import pandas as pd
from tqdm import tqdm
index_set=pd.read_csv("index_set.csv",index_col=0)["0"]
df=pd.DataFrame(index=index_set)
file_list = os.listdir("tmp2")
for i in tqdm(file_list):
    a=pd.read_csv("tmp2/"+i,index_col=0)
    b=list(a.index)[0]
    df[i] = a.loc[b]
df=df.fillna(0)
df.to_csv("split_total.csv")