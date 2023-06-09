# -*- coding: utf-8 -*-
"""Untitled78.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/188N0ysWzmh9GlkxhAcqEDn4nD-tXmhtn
"""

import numpy as np
import pandas as pd

path='/content/2020_가구마스터_20230320_21294.csv'
df = pd.read_csv(path, encoding='CP949')
columns = ["조사연도", '수도권여부', 'MD제공용_가구고유번호', '가구주_성별코드', '가구주_만연령', '가구원수', '가구주_교육정도_학력코드', '가구주_혼인상태코드', '순자산', '부채', '처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]', '가구주_산업대분류코드', '가구주_직업대분류코드', '입주형태코드']
df1= df[columns]
df1.rename(columns={'조사연도':"year",
'수도권여부':"metro",
'MD제공용_가구고유번호':"id",
'가구주_성별코드':"sex",
'가구주_만연령':"age",
'가구원수':"number",
'가구주_교육정도_학력코드':"education",
'가구주_혼인상태코드':"marriage",
'순자산':"asset",
'부채':"debt",
'처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]':"income_d",
'가구주_산업대분류코드':"industry",
'가구주_직업대분류코드':"job",
'입주형태코드':"house"}, inplace=True)

df1.tail()

"""## 2020년 가계금융복지조사를 이용한 우리나라 전체가구 평균소득 구하기"""

income = df1['income_d'].dropna()
# income.describe()
se = np.std(income) / np.sqrt(len(income))
np.mean(income), np.round(np.std(income), 2), se

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,6))
ax.hist(income, alpha = 0.3, bins = 2000)
ax.grid(True)
ax.set_xlim((-5000, 20000))
plt.show()

"""## 500가구 10000번 무작위 추출 후 평균 구하기
### 히스토그램으로
"""

n=10000
mn =[]

for i in range(n):
    sample = np.random.choice(income, size = 500, replace=True)
    mean = np.mean(sample)
    mn.append(mean)

fig,ax = plt.subplots(figsize = (15,6))
ax.hist(mn, alpha=0.3, bins=1000)
ax.grid()
ax.set_xlim((3500, 5500))
plt.show()

"""## 가구 평균소득에 대한 신뢰구간
- 95%신뢰구간 구해보기
"""

n = 1000

interval = np.zeros((n, 2))

for i in range(n):
    sample=np.random.choice(income, size=500, replace=True)
    se = np.std(sample) / np.sqrt(len(sample))
    mn = np.mean(sample)
    lb = mn-2*se
    hb = mn+2*se
    interval[i-1][0] = lb
    interval[i-1][1] = hb

df1= pd.DataFrame(data=interval, columns=['lb', 'hb'])
df1['mean'] = np.mean(income)
df1['test_l'] = np.where((df1['mean'] < df1['lb']), 1, 0)
df1['test_h'] = np.where((df1['mean'] > df1['hb']), 1, 0)
df1['test_t'] = df1['test_l'] +df1['test_h']
df1.head()

# 분석 결과 100번의 시행 중 신뢰구간이 모평균을 포함하지 않는 경우 = 47
df1['test_t'].sum()

