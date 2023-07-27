import pandas as pd
import jieba.posseg as pseg
import os
from tqdm.notebook import tqdm
import feather
import numpy as np

##基于总词库分词（改进）——>生成feather形式的向量
def to_vector(n1,n2):
    ##定义构建归一化csv文档的函数
    def normalize(frame_data,path):
        vectors ={}
        for name, data in tqdm(frame_data.items(),desc = "normalize",leave = False):
            frame = pd.DataFrame.from_dict(data, orient='index', columns=[name])
            vector = np.array(frame[name])
            norm = np.linalg.norm(vector)
            vector = vector / norm
            vectors[name] = vector
        vectors=pd.DataFrame(vectors)
        feather.write_dataframe(vectors,path)

    ##调用词库    
    index = pd.read_csv("词库构建/word2.csv")
    index = set(index["word"])

    ##开始分词
    for year in tqdm(range(n1,n2),desc="Year"):
        folder_path = str(year)  
        txt_path = os.path.join("TXT",folder_path)
        file_list = os.listdir(txt_path)  # 获取文件夹中的所有文件名

        frame_data = {}  # 用字典存储各个 frame 数据
        IDF = {word: 0 for word in index} #用字典储存IDF值

        for file_name in tqdm(file_list, desc="First loop",leave = False):    

            number1 = file_name[:-4]
            file_path = os.path.join(txt_path, file_name)
            IDF_add = {}   #储存需要加的IDF

            number_word = 0
            frame_data[number1] = {word: 0 for word in index}

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            content = pseg.cut(content)

            for word, flag in content:    
                if not flag.startswith('x'):
                    number_word += 1
                    if word in index:
                        frame_data[number1][word] += 1
                        if word not in IDF_add:
                            IDF_add[word] = 1
            del content
            ##计算TF
            for key , value in frame_data[number1].items():
                frame_data[number1][key] = value/number_word

            ##合并IDF(原始)
            for key , value in IDF_add.items():
                IDF[key] += value

            del IDF_add

        #计算IDF
        N = len(file_list)
        for key , value in tqdm(IDF.items(),desc = "IDF",leave = False):
            IDF[key] = np.log(N/(1+value))

        #计算TF-IDF
        for key1 in tqdm(frame_data.keys(),desc = "TF-IDF",leave = False):
            for key2 , value2 in frame_data[key1].items():
                frame_data[key1][key2] = value2*IDF[key2]
        del IDF

        ##创建文件夹
        sum_path = os.path.join("sum",folder_path)
        try:
            os.makedirs(sum_path)
        except:
            pass
        
        feather_path = os.path.join(sum_path,folder_path+"_vector.feather")
        # 构建归一化csv文档
        normalize(frame_data,feather_path)        
        del frame_data


##构建相似度矩阵
def to_matrix(n1,n2):
    for year in tqdm(range(n1,n2),desc = "Year"):
        txt_path = os.path.join("sum",str(year))
        vectors = pd.read_feather(os.path.join(txt_path,str(year)+"_vector.feather"))
        file_list = vectors.columns
        num_files = len(file_list)
        
        # 创建矩阵来存储相似度矩阵
        similarity = np.zeros((num_files, num_files))

        # 计算相似度矩阵
        for i in tqdm(range(num_files),desc = "Outer loop",leave = False):
            vector1 = vectors[file_list[i]]

            for j in tqdm(range(i+1, num_files),desc ="Inner loop",leave = False):
                vector2 = vectors[file_list[j]]
                similarity[i, j] = np.dot(vector1, vector2)
                similarity[j, i] = similarity[i, j]

            similarity[i, i] = 1

        # 将相似度矩阵转换为DataFrame
        similarity_df = pd.DataFrame(similarity, index=file_list, columns=file_list)

        # 保存相似度矩阵为Feather文件
        feather.write_dataframe(similarity_df,os.path.join(txt_path,str(year)+"_matrix.feather"))


##生成n个行业分类
def process_similarity(n1,n2,number):
    
    def classification(frame,path):        
        names=frame.columns.tolist()
        dict1={}
        for a,element in enumerate(tqdm(names)):
            name=element.split("+")
            name = ["\t"+x for x in name]
            a=str(a)+"_industry"
            dict1[a]=name
        a=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dict1.items()]))
        a.to_csv(path)
        
    for year in tqdm(range(n1,n2),desc = "Year"):
        txt_path = os.path.join("sum",str(year))
    
        similarity_origin=pd.read_feather(os.path.join(txt_path,str(year)+"_matrix.feather"))
        similarity=similarity_origin.copy()
        np.fill_diagonal(similarity.values, np.nan)

        for i in tqdm(range(len(similarity) - number),desc="Outer loop",leave = False):

            a,b=similarity.stack().idxmax()
            similarity.drop(index=[a,b],columns=[a,b],inplace=True)
            new_name = a + "+" + b 
            count1 = new_name.split("+")
            del a
            del b

            for index in tqdm(similarity.index,desc ="Inner loop",leave = False):
                df=0
                count2 = index.split("+")
                for m1 in count1:
                    for m2 in count2:
                        df+=similarity_origin.at[m1,m2]
                similarity.at[index,new_name] = df/(len(count1) * len(count2))
                similarity.at[new_name,index] = similarity.loc[index,new_name]
                del m1
                del m2 
                del count2
            similarity.loc[new_name,new_name]=np.nan 
            del count1
            del new_name

        similarity=similarity.fillna(1)
        classification(similarity,os.path.join(txt_path,str(year)+"_"+str(number)+"industry.csv"))



###寻找同行
def competitors(n1,n2,number):
    for year in tqdm(range(n1,n2),desc = "Year"):
        txt_path = os.path.join("sum",str(year))
        matrix=pd.read_feather(os.path.join(txt_path,str(year)+"_matrix.feather"))
        industry = pd.read_csv(os.path.join(txt_path,str(year)+"_"+str(number)+"industry.csv"),index_col = 0,dtype="str")
        competitors = {}
        for company in tqdm(matrix.columns,desc = "competitors"):
            
            same_industry = set()
            flag = 0
            
            #找同行业成员
            for i in industry.columns:
                for j in industry[i]:
                    if pd.notna(j) and company == j[-6:]:
                        same_industry.update([x[-6:] for x in industry[i] if pd.notna(x)])
                        flag = 1
                        break
                if flag == 1:
                    break
            
            #计算阈值
            threshold = sorted([matrix.at[company, x] for x in same_industry])[0]
            sorted_col = matrix[company].sort_values(ascending=False).iloc[1:,]
            competitors["\t"+company] = list("\t"+x for x in sorted_col.index if sorted_col.at[x] >= threshold)
        competitors = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in competitors.items()]))
        competitors.to_csv(os.path.join(txt_path,str(year)+"_"+str(number)+"based_competitor.csv"))
    

