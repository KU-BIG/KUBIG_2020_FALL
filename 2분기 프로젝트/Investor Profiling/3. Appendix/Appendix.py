#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


act = pd.read_csv('2_act_info.csv')
cus = pd.read_csv('2_cus_info.csv')
iem = pd.read_csv('2_iem_info.csv')
trd_kr =  pd.read_csv('2_trd_kr.csv')
trd_oss =  pd.read_csv('2_trd_oss.csv')
cus = cus.astype({'tco_cus_grd_cd' : 'str'})
trd_kr['orr_dt'] = pd.to_datetime(trd_kr['orr_dt'].astype(str), format='%Y%m%d')
trd_oss['orr_dt'] = pd.to_datetime(trd_oss['orr_dt'].astype(str), format='%Y%m%d')
cus_info = pd.read_csv('cus_info.csv')


# # 5. Appendix

# ## 5.1 타당성 검정

# In[4]:


from statsmodels.stats.proportion import proportions_ztest
import numpy as np

cus_info_20 = cus_info.loc[cus_info['age_cat']=='20대']
low_20 = cus_info_20.loc[(cus_info_20['grade']=='09 등급없음') | (cus_info_20['grade']=='05 블루') | (cus_info_20['grade']=='04 그린')]
high_20 = cus_info_20.loc[(cus_info_20['grade']=='03 로얄') | (cus_info_20['grade']=='02 골드') | (cus_info_20['grade']=='01 탑클래스')]

count = np.array([low_20['grade'].count(), high_20['grade'].count()])
nobs = np.array([cus_info_20['grade'].count(), cus_info_20['grade'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(z)
print(p)


# 편의상 20대를 두 등급으로(적은 자산규모/큰 자산규모) 분리했습니다.   
# 두 그룹간의 비율의 차이가 통계적으로 유의미한지 검정하기위해 정규분포를 이용한 proportions_ztest를 사용했습니다.     
# 통계적 유의성의 척도인 p-value가 p변수에 나타나 있습니다. p-value가 0에 가까우므로 두 그룹간의 비율이 동일하다는 영 가설을 기각할 수 있습니다.

# In[5]:


cus_info_30 = cus_info.loc[cus_info['age_cat']=='30대']
low_30 = cus_info_30.loc[(cus_info_30['grade']=='09 등급없음') | (cus_info_30['grade']=='05 블루') | (cus_info_30['grade']=='04 그린')]
high_30 = cus_info_30.loc[(cus_info_30['grade']=='03 로얄') | (cus_info_30['grade']=='02 골드') | (cus_info_30['grade']=='01 탑클래스')]

count = np.array([low_30['grade'].count(), high_30['grade'].count()])
nobs = np.array([cus_info_30['grade'].count(), cus_info_30['grade'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(z)
print(p)


# 30대도 위와 동일한 과정을 거쳤고, 같은 결과가 나왔습니다.     
# 즉 2,30대 모두 작은 규모의 자산을 가지고 주식투자를 하고 있다는 주장의 타당성을 통계적으로 검정하였습니다.

# In[6]:


cus_info_40 = cus_info.loc[cus_info['age_cat']=='40대']
cus_info_50 = cus_info.loc[cus_info['age_cat']=='50대']
cus_info_60 = cus_info.loc[cus_info['age_cat']=='60대']
cus_info_70 = cus_info.loc[cus_info['age_cat']=='70대']

risk_20 = cus_info_20.loc[(cus_info_20['ivst']=='05 공격투자형') | (cus_info_20['ivst']=='04 적극투자형')]
risk_30 = cus_info_30.loc[(cus_info_30['ivst']=='05 공격투자형') | (cus_info_30['ivst']=='04 적극투자형')]
risk_40 = cus_info_40.loc[(cus_info_40['ivst']=='05 공격투자형') | (cus_info_40['ivst']=='04 적극투자형')]
risk_50 = cus_info_50.loc[(cus_info_50['ivst']=='05 공격투자형') | (cus_info_50['ivst']=='04 적극투자형')]
risk_60 = cus_info_60.loc[(cus_info_60['ivst']=='05 공격투자형') | (cus_info_60['ivst']=='04 적극투자형')]
risk_70 = cus_info_70.loc[(cus_info_70['ivst']=='05 공격투자형') | (cus_info_70['ivst']=='04 적극투자형')]


# In[7]:


count = np.array([risk_20['ivst'].count(), risk_30['ivst'].count()])
nobs = np.array([cus_info_20['ivst'].count(), cus_info_30['ivst'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_20['ivst'].count(), risk_40['ivst'].count()])
nobs = np.array([cus_info_20['ivst'].count(), cus_info_40['ivst'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_20['ivst'].count(), risk_50['ivst'].count()])
nobs = np.array([cus_info_20['ivst'].count(), cus_info_50['ivst'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_20['ivst'].count(), risk_60['ivst'].count()])
nobs = np.array([cus_info_20['ivst'].count(), cus_info_60['ivst'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)


# 각 연령대 그룹을 편의상 크게 안정지향/위험지향으로 나누고     
# 20대와 나머지 연령대간의 위험지향 비율이 같은지 (영 가설), 20대의 위험지향 비율이 더 낮은지(대립가설)를 통계적으로 검정했습니다.     
# 이 경우 단측검정을 사용하기 때문에 구한 p-value값을 2로 나눕니다.     
# p-value값이 0.01보다 작으므로 영 가설을 기각하고 대립가설을 채택할 수 있습니다.     
# 즉, 20대의 위험지향 비율이 나머지 연령대들보다 낮다는 것을 통계적으로 입증하였습니다

# In[8]:


top = cus_info.loc[cus_info['grade']=='01 탑클래스']
gold = cus_info.loc[cus_info['grade']=='02 골드']
royal = cus_info.loc[cus_info['grade']=='03 로얄']
green = cus_info.loc[cus_info['grade']=='04 그린']
blue = cus_info.loc[cus_info['grade']=='05 블루']

risk_top = top.loc[top['risk']=='03 위험감수형']
risk_gold = gold.loc[gold['risk']=='03 위험감수형']
risk_royal = royal.loc[royal['risk']=='03 위험감수형']
risk_green = green.loc[green['risk']=='03 위험감수형']
risk_blue = blue.loc[blue['risk']=='03 위험감수형']


# In[9]:


count = np.array([risk_top['risk'].count(), risk_gold['risk'].count()])
nobs = np.array([top['risk'].count(), gold['risk'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_gold['risk'].count(), risk_royal['risk'].count()])
nobs = np.array([gold['risk'].count(), royal['risk'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_royal['risk'].count(), risk_green['risk'].count()])
nobs = np.array([royal['risk'].count(), green['risk'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)

count = np.array([risk_green['risk'].count(), risk_blue['risk'].count()])
nobs = np.array([green['risk'].count(), blue['risk'].count()])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(p/2)


# 영 가설은 (탑클래스/골드), (골드/로얄), (로얄/그린), (그린/블루)간의 위험감수형 비율 차이가 없다이고,    
# 대립가설은 (탑클래스/골드), (골드/로얄), (로얄/그린), (그린/블루)에서 전자의 위험감수형 비율이 후자의 위험감수형 비율보다 높다입니다.  
# 
# p-value값들이 전부 0.05보다 작으므로 95%의 신뢰수준하에서 영 가설을 기각할 수 있습니다. 
# 즉 탑클래스가 골드보다, 골드가 로얄보다, 로얄이 그린보다, 그린이 블루보다 위험을 감수하는 경향이 높다고 할 수 있습니다.
# 
# (탑클래스/골드)의 경우 p-value가 0.0366으로 유독 큰데 이는 신뢰수준을 99%로 높게 잡는다면 영 가설을 기각할 수 없습니다.     
# 하지만 보통 통계적 타당성의 유의성 기준을 p-value<0.05로 잡으므로 신뢰수준 95%하에서는 영 가설을 기각할 수 있다.    
# 실제로 두 그룹간의 위험감수형 비율차이는 타 그룹들과 비교하면 매우 작은 것을 눈으로도 쉽게 확인할 수 있습니다.

# p-value가 0.01보다 작으므로 20,30대의 보유 계좌개수와 40대의 보유 계좌개수 사이에는 유의미한 차이가 있습니다.

# ## 5.2 참고자료

# [종목별 산업 종류 데이터](https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage)
# 
# 
# [기업 시가총액 데이터](http://marketdata.krx.co.kr/mdi#document=undefined)
# 
# 
# [최근 개인투자자 주식 매수의 특징 및 평가](https://eiec.kdi.re.kr/policy/domesticView.do?ac=0000153203)
