# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:40:36 2019

@author: ZuoYS
"""

path = 'D:/课程-2019秋季/大数据分析/Assignment1-NMT or Recommendation/data_multi30k&newstest2014/'
eng_set = 'train.en.txt'
de_set = 'train.de.txt'
target_set = 'eng-de.txt'
with open(path + eng_set, 'r', encoding = 'utf-8') as f:
    line_eng = [line.strip() for line in f.readlines()]
    #line_eng = f.readlines().split("\n")
f.close()

with open(path + de_set,'r',encoding = 'utf-8') as g:
    line_de = [line.strip() for line in g.readlines()]
    #line_de = g.readlines().split("\n")
g.close()

new_set_list = []

for i in range(len(line_eng)-1):
    new_set_list.append(line_eng[i]+'\t'+line_de[i])

with open(path + target_set, 'w', encoding = 'utf-8') as f_target:
    for item in new_set_list:
        f_target.write("%s\n" % item)

f_target.close()