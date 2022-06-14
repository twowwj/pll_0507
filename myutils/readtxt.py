
import numpy as np
from collections import defaultdict

def txt2array(txt_path, delimiter):
    #---
    # 功能：读取只包含数字的txt文件，并转化为array形式
    # txt_path：txt的路径；delimiter：数据之间的分隔符
    #---
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(delimiter)
        temp = list(map(str, data_split))
        data_list.append(temp)

    data_array = np.array(data_list)
    return data_array


path = '/data/wjwang/pll_0507/rafdb/test.txt'
data_list= txt2array(path, ',')[1]

dict_final ={}
for i in range(len(data_list)):
    key = data_list[i].split(":")[0]
    value = data_list[i].split(":")[1]
    dict_final[key] = value



def top_N(d, n):
    dct= defaultdict(list)
    for k, v in d.items():
        dct[v].append(k)
    return sorted(dct.items())[-n:][::-1]
aa = top_N(dict_final, 19)

print(aa)