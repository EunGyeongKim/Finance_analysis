#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager, rc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


path = '/usr/local/lib/python3.9/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family = font_name)


# In[6]:


path = '/content/2020_가구마스터_20230320_21294.csv'
df = pd.read_csv(path, encoding='CP949')


# In[7]:


columns = ['조사연도', '수도권여부', 'MD제공용_가구고유번호', '가구주_성별코드', '가구주_만연령', '가구원수', '가구주_교육정도_학력코드', '가구주_혼인상태코드', '순자산', '부채', '처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]', '가구주_산업대분류코드', '가구주_직업대분류코드', '입주형태코드']
df = df[columns].copy()
df.rename(columns={'조사연도':'year','수도권여부':'metro','MD제공용_가구고유번호':'id','가구주_성별코드':'sex','가구주_만연령':'age','가구원수':'number','가구주_교육정도_학력코드':'education','가구주_혼인상태코드':'marriage','순자산':'asset','부채':'debt','처분가능소득(보완)[경상소득(보완)-비소비지출(보완)]':'income','가구주_산업대분류코드':'industry','가구주_직업대분류코드':'job','입주형태코드':'house'}, inplace=True)


# In[8]:


# 30대 1인가구
df1 = df.loc[df['number'].isin([1]) & (df['age']>30) & (df['age']<40) ]
df2 = df1[['sex', 'number', 'age', 'income']]


# In[11]:


import seaborn as sns
df21 = df2.loc[df2['sex'].isin([1])] # 남
df22 = df2.loc[df2['sex'].isin([2])] # 여

sns.kdeplot(df21['income'], shade=True, label='male', clip=(-1000,20000))
sns.kdeplot(df22['income'], shade=True, label='female', clip=(-1000,20000))
plt.xlabel('10,000 won')
plt.legend()
plt.grid(True)
plt.show()


# ## 유의성 검정
# z-통계량 구하기
# 

# In[12]:


df3 = df2[['income']].groupby(df2['sex']).agg(['mean', 'std', 'count'])
df3


# In[13]:


mean = df3[('income', 'mean')]
mean_df = mean[1]-mean[2]
np.round(mean_df, 2)


# In[15]:


std = df3[('income', 'std')]
count = df3[('income', 'count')]
se1 = std[1]/np.sqrt(count[1])
se2 = std[1]/np.sqrt(count[1])
tot_se = np.sqrt(se1**2 + se2**2)
np.round(tot_se, 2)


# In[16]:


z = mean_df / tot_se
round(z, 2)


# In[17]:


import scipy as sp
import scipy.stats

rv = sp.stats.norm(loc=0, scale=1) # 평균 0, 표준편차1, 표준정규분포
np.round(1-rv.cdf(z), 2)


# p-value = 0.03. 따라서 5%의 유의수준에서 귀무가설(30대 남성 1인 가구 소득과 30대 여성 1인 가구와의 소득차이는 없다)는 기각됨.
