# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:21:27 2023

@author: a
"""
from pandas import DataFrame
import jieba.posseg as pseg
import os

folder_path = '2007'  # 替换为你的文件夹路径
file_list = os.listdir(folder_path)  # 获取文件夹中的所有文件名

names = ["2007/"+ x for x in file_list]
list1_summary=[]
final_count={} 

n=0
example_dict={}
for i in names[n:]:
    a=open(i,"r",encoding="utf-8",errors='ignore')
    A=a.read()
###读取全部内容

    b=pseg.cut(A,HMM=True)
    b1=[]
    for word,flag in b:
        if flag.startswith('n'):
            b1.append(word)
###只要名词     

    b2 = [word for word in b1 if len(word) >= 2]       
###只报告两个汉字以上的内容

    count=example_dict.copy()
    for item in b2:
        if item in count:
            count[item]+=1  
        else: 
            count[item]=1             
    number1=i[5:-4]
    df=DataFrame(count,index=[number1])
    df.to_csv("tmp2/"+number1+".csv")