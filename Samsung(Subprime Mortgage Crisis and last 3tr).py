# -*- coding: utf-8 -*-
"""# 주가수익률"""

!pip install finance-datareader

import numpy as np

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

import pandas as pd
import FinanceDataReader as fdr

# 삼전 주가 불러오기 (리먼사건으로 인한 경제위기 기간 vs 최근 3년)
df = fdr.DataReader('005930', '2007-01-01')  

recent_df = df.loc['2020-03-20':]    #최근3년 
df = df.loc['2007-06-01':'2009-06-30']     #경제위기기간


# recent_df describe
r = (df['Close'].pct_change())*100
r.describe()

#최근 3년 describe
r2 = (recent_df['Close'].pct_change())*100
r2.describe()

# 일별 수익률 그래프
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
ax[0].plot(r, linestyle='', marker='o', alpha=0.7, ms=3)
ax[0].vlines(r.index, 0, r.values, lw = 0.2)
ax[0].set_ylabel('return(%)', fontsize = 12)
ax[0].set_xlabel('date', fontsize =12)
ax[0].grid()

ax[1].plot(r2, linestyle='', marker='o', alpha=0.7, ms=3)
ax[1].vlines(r2.index, 0, r2.values, lw = 0.2)
ax[1].set_ylabel('return(%)', fontsize = 12)
ax[1].set_xlabel('date', fontsize =12)
ax[1].grid()
plt.show()

# 일별 수익률을 히스토그램으로
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 6))
ax[0].hist(r, alpha=0.3, bins = 200)
ax[0].grid()
ax[0].set_xlim((-10, 10))

ax[1].hist(r2, alpha=0.3, bins = 200)
ax[1].grid()
ax[1].set_xlim((-10, 10))
plt.show()

# 실제 수익률 데이터와 동일한 평균과 분산을 가지고 있는 가상데이터 정규분포 생성
# 그후 3년치 데이터와 비교

mu = r2.mean()
s = r2.std()
data = np.random.randn(len(r2))*s+mu

fig, ax = plt.subplots(figsize =(10, 6) )
ax.hist([r], alpha=0.3, bins=100, density=True, color='blue', label='economic crisis')
ax.hist([data], alpha=0.3, bins=100, density=True, color='red', label='last 3yr')
ax.set_xlim((-5, 5))
ax.grid(True)
plt.legend()
plt.show()

"""# 주가 등락을 이항분포로 설명하기"""

import math

#lsat 3yr
df2 = recent_df
df2['ret'] = (df2['Close'].pct_change())*100 # 일별 수익률 게산
df3 = df2['ret']
r_3yr = df3.dropna() # 거래일수

positive_3yr = 0
n_3yr = len(r_3yr)

for i in range(n_3yr):
    if r_3yr[i]>0:
        positive_3yr += 1
negative = n_3yr-positive_3yr

p_rate_3yr = positive_3yr / n_3yr # 주가 상승 비율 0.46756756756756757
len(r_3yr), positive_3yr, p_rate_3yr # 전체 일수, 주가 상승일, 상승률
#(740, 346, 0.46756756756756757)

# 다음날 주가가 오를지, 내릴지 베르누이 확률변수로 표현 가능 -> 이항분포로 설명가능
# 주식가격이 오르는 날 수 -> X
# X~B -> 0.46393762183235865
# 따라서 5일중 주식가격이 3일 오를 확률은 다음과 같음
# p(X=3) = (5!) / (3! * (5-3)!) * (1-0.46393762183235865)^2 = 28.6%
# 코드로 표현
a = math.factorial(5)
b = math.factorial(3) * math.factorial(2)
c = ((p_rate_3yr)** 3 ) * ((1-p_rate_3yr)**2)
number_3yr = a/b

prob_3yr = number_3yr * c
prob_3yr

#economic crisis
df2 = df.copy()
df2['ret'] = (df2['Close'].pct_change())*100 # 일별 수익률 게산
df3 = df2['ret']
r_cris = df3.dropna() # 거래일수

positive_cris = 0
n_cris = len(r_cris)

for i in range(n_cris):
    if r[i]>0:
        positive_cris += 1
negative_cris = n_cris-positive_cris

p_rate_cris = positive_cris / n_cris # 주가 상승 비율  0.46393762183235865
len(r_cris), positive_cris, p_rate_cris # 전체 일수, 주가 상승일,상승률
#(513, 238, 0.46393762183235865)

# 다음날 주가가 오를지, 내릴지 베르누이 확률변수로 표현 가능 -> 이항분포로 설명가능
# 주식가격이 오르는 날 수 -> X
# X~B -> 0.46393762183235865
# 따라서 5일중 주식가격이 3일 오를 확률은 다음과 같음
# p(X=3) = (5!) / (3! * (5-3)!) * (1-0.46393762183235865)^2 = 28.6%
# 코드로 표현
a = math.factorial(5)
b = math.factorial(3) * math.factorial(2)
c = ((p_rate_cris)** 3 ) * ((1-p_rate_cris)**2)
number_cris = a/b

prob_cris = number_cris * c
prob_cris

"""## 합의 표준오차
- 2021년 삼전 주가가 513일중 238일이 상승
- 삼전의 주가가 1년중 오를 변동성(확률오차) 구하기
- 삼전의 주가가 오르는 날의 표준오차로 나타낼 수 있음
- 합의 표준오차 = 상자의 표준편차 * root(추출횟수)
- 합의 표준오차를 구하기 위하서 먼저 상자모형을 설정 -> 상자의 표준오차 구하기

"""

# 합의 표준오차 economic crisis
se_cris = np.sqrt(p_rate_cris*(1-p_rate_cris))* np.sqrt(n_cris)
se_cris

# 합의 표준오차 last 3yr
se_3yr = np.sqrt(p_rate_3yr*(1-p_rate_3yr))* np.sqrt(n_3yr)
se_3yr

"""# 비율의 표준오차
- 삼전의 주가가 2021년 거래일 중에서 오를 확률은 44.5%
- 이떄 삼전의 주가가 오를 비율의 변동성(확률오차)를 다음 공식을 이용해 구해보자
    - 평균의 표준오차 = 상자의 표준편차 / root(추출횟수)
    - 상자의 표준오차 = 0.4964
    - 1년중 거래 일자 244
    - 그러므로 표준오차는 약 3.2% (0.4964 / root(244))
    - 삼전으 44.5%비율 주변으로 약 3% 전후로 오름
"""

se2_cris = np.sqrt(p_rate_cris * (1-p_rate_cris)) / np.sqrt(n_cris)
se2_cris * 100

se2_3yr = np.sqrt(p_rate_3yr * (1-p_rate_3yr)) / np.sqrt(n_3yr)
se2_3yr * 100

"""# 주식투자 시 최대 예상 손실액 구하기
- 1억원을 투자했을 때 6개월(125 영업일 가정) 보유할 경우 발생가능한 최대 손실액(95% VaR:Value of risks)구하기
    - 최대 손실액 : 정상적인 시장 여건 하에서 일정기간동안 발생할 수 있는 '최대손실금액'  
    ex) 신뢰수준 95%, 보유기간 6개월에서 산출된 VaR가 1억원  
        -> 해당 자산의 가치 변동으로 인해 6개월동안 발생할수 있는 손실금액이 1억원을 넘지 않을 확률이 95%,   
        1억원을 초과할 확률이 5%라는 의미  
- <최대 예산 손실액 구하는 방법>
1. 과거 10년간의 일별 수익률 표본에서 6개월(125개)의 일별 수익률을 무작위 복원 추출하여 부트스랩 표본 평균 수익 구하기(1000번)
2. 이러한 확률히스토그램의 상위 95%(하위 5%)에 해당하는 수익률에 1억원을 곱한 금액이 95% 유의수준에서 최대 예상 손실액
3. 삼전 로그수익률 구하기
    - 로그 수익률 : 연속복리 수익률 의미
        - 일반적인 수익률 : 플러스 수익률은 무한대의 범위를 갖는 반면, 마이너스 수익률은 -100%까지 범위를 가지게 됨.
        - 특정기간에 로그수익률은 매기의 로그 수익률을 더하는것과 같음.
"""

# 삼전 주가 불러오기 
# df = fdr.DataReader('005930', '2007-06-01')

df['ret'] = (np.log(df.Close) - np.log(df.Close.shift(1))) # 로그 수익률 
df1 = df.loc['2007-06-01':'2009-06-30']        
df2 = df1.dropna()
df2 = df2['ret']
len(df2)

recent_df['ret'] = (np.log(recent_df.Close) - np.log(recent_df.Close.shift(1))) # 로그 수익률 
recent_df2 = recent_df.dropna()
recent_df2 = recent_df2['ret']


len(df2), len(recent_df2)

#125개(6month)의 로그수익률을 무작위로 복원추출하여 합산 로그수익률 구하기
# 이 과정 1000번 반복
sum_boots_cris = []
sum_boots_3yr = []

day = 125
n= 1000

for i in range(n):
    sample = np.random.choice(df2, size=day, replace=True)
    sample_sum = sample.sum()
    sum_boots_cris.append(sample_sum)

for i in range(n):
    sample2 = np.random.choice(recent_df2, size=day, replace=True)
    sample_sum = sample2.sum()
    sum_boots_3yr.append(sample_sum)

#1000개 hist
fig, ax = plt.subplots(figsize =(8,5))
ax.hist(sum_boots_cris, alpha=0.3, bins=100, color='blue', label='economic crisis')
ax.hist(sum_boots_3yr, alpha=0.3, bins=100, color='red', label='last 3yr')
ax.grid(True)
ax.set_xlim((-1,1))
plt.legend()
plt.show()

# crisis
# 1억원을 투자하여 6개월 보유했을때 95% VaR로 계산한 최대 예상 손실액
ret = np.array(sum_boots_cris)
ret5 = np.percentile(ret, 5)
lower_ret5 = np.exp(ret5) -1 # 로그 수익률을 일반수익률로 변환
lower_ret5 # 하위 5% 일반 수익률

var95 = ret5*100000000
var95 # 95% VaR로 계산한 최대 예상 손실액

# 3yr
# 1억원을 투자하여 6개월 보유했을때 95% VaR로 계산한 최대 예상 손실액
ret = np.array(sum_boots_3yr)
ret5 = np.percentile(ret, 5)
lower_ret5 = np.exp(ret5) -1 # 로그 수익률을 일반수익률로 변환
lower_ret5 # 하위 5% 일반 수익률

var95 = ret5*100000000
var95 # 95% VaR로 계산한 최대 예상 손실액











