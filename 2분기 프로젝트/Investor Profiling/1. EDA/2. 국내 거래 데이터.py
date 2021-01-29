#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "notebook_connected"
import warnings
warnings.filterwarnings("ignore")

from pyecharts.charts import Bar, Grid, Line, Liquid, Page, Pie, Timeline
from pyecharts import options as opts


pd.options.display.float_format = '{:.2f}'.format
col = px.colors.qualitative.Pastel


# In[9]:


act = pd.read_csv('2_act_info.csv')
cus = pd.read_csv('2_cus_info.csv')
iem = pd.read_csv('2_iem_info.csv')
trd_kr =  pd.read_csv('2_trd_kr.csv')
trd_oss =  pd.read_csv('2_trd_oss.csv')
cus = cus.astype({'tco_cus_grd_cd' : 'str'})
trd_kr['orr_dt'] = pd.to_datetime(trd_kr['orr_dt'].astype(str), format='%Y%m%d')
trd_oss['orr_dt'] = pd.to_datetime(trd_oss['orr_dt'].astype(str), format='%Y%m%d')
act_cus = pd.read_csv('act_cus.csv')
cus_info = pd.read_csv('cus_info.csv')


# ## 1.2 Korea Trading Data

# In[10]:


trd_kr = pd.merge(act_cus, trd_kr, on='act_id') #거래-고객 정보 데이터
trd_kr['amount'] = trd_kr['cns_qty']*trd_kr['orr_pr'] #거래별 금액 계산


# In[11]:


#고객별 종목 개수

a2 = dict(list(trd_kr[['cus_id','iem_cd']].groupby('cus_id')))
b2 = trd_kr.cus_id.unique().tolist()
e2 = []

for i in b2:
  c2 = a2[i]
  d2 = len(c2.iem_cd.unique())
  e2.append(d2)

df = pd.DataFrame({'cus_id': b2, 'iem_num_kr': e2})
cus_info = pd.merge(cus_info, df, on='cus_id',how='outer')


# In[12]:


kr = trd_kr.rename({'orr_rtn_hur':'ord','lst_cns_hur':'real','sby_dit_cd':'buysell', 
                        'orr_mdi_dit_cd':'store', 'cns_qty':'qty', 'orr_pr':'price'},axis='columns')


# 고객별 국내 투자 종목 개수를 뽑아 **iem_num_kr** 변수를 만듭니다.

# In[13]:


df = kr

#매도, 매수 데이터 분할
df_sell = df.loc[df['buysell']==1]
df_buy = df.loc[df['buysell']==2]
df_sell = df_sell.groupby(by=['act_id','iem_cd'], as_index=False)
df_buy = df_buy.groupby(by=['act_id','iem_cd'], as_index=False)

#종목별 매도/매수 주식개수 총합, 거래액 총합 계산
act_sell = df_sell.sum()[['act_id','iem_cd','qty','amount']].fillna(0)
act_buy = df_buy.sum()[['act_id','iem_cd','qty','amount']].fillna(0)

act_sell.rename(columns={'amount' : 'sell_amount_kr', 'qty':'sell_qty_kr'}, inplace=True)
act_buy.rename(columns={'amount' : 'buy_amount_kr', 'qty':'buy_qty_kr'}, inplace=True)

#종목별 매도/매수 건수 계산
count_sell = df_sell.count()[['act_id','iem_cd','buysell']].fillna(0)
count_buy = df_buy.count()[['act_id','iem_cd','buysell']].fillna(0)
count_sell.rename(columns={'buysell' : 'sell_num_kr'}, inplace=True)
count_buy.rename(columns={'buysell' : 'buy_num_kr'}, inplace=True)

#구한 값들을 합쳐줍니다.
df = pd.merge(act_sell, act_buy, on=['act_id','iem_cd'],how="outer")
df = pd.merge(df, count_sell, on=['act_id','iem_cd'],how="outer")
df = pd.merge(df, count_buy, on=['act_id','iem_cd'],how="outer")
trdkr = df.fillna(0) #한국: 계좌 - 종목별 매도/매수 건수, 총합 들어있는 데이터
trdkr['total_num_kr'] = trdkr['buy_num_kr'] + trdkr['sell_num_kr']


# In[14]:


#계좌별 매도/매수 건수 계산
trdkr2 = trdkr.groupby(trdkr['act_id']).sum()

#고객별 매도/매수 건수 계산
trdkr3 = pd.merge(trdkr2, act_cus, on=['act_id'])
trdkr4 = trdkr3.groupby(trdkr3['cus_id']).sum()
cus_info = pd.merge(trdkr4, cus_info, on=['cus_id'], how='outer')


# 종목별 국내 매도, 매수 주식 개수를 추출해 **sell_qty_kr**, **buy_qty_kr** 변수를 만듭니다.    
# 종목별 국내 매도, 매수 주식 개수를 추출해 **sell_amount_kr**, **buy_amount_kr** 변수를 만듭니다.    
# 종목별 국내 매도, 매수 주식거래 건수를 추출해 **sell_num_kr**, **buy_num_kr** 변수를 만들고, 둘을 합쳐 **total_num_kr** 변수를 만듭니다.

# In[15]:


#국내 거래 건수 시각화

df = cus_info.groupby(by=['age_cat']).median().reset_index()

trace3 = go.Bar(x=df.age_cat, y=df.sell_num_kr, name='매도',marker_color=col[0])
trace4 = go.Bar(x=df.age_cat, y=df.buy_num_kr, name='매수',marker_color=col[3])

data = [trace3, trace4]

layout = go.Layout(title='고객별 국내 거래 건수(중앙값)',barmode='stack')

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# 연령대별 매수, 매도 건수를 시각화 해보았습니다.
# 
# 우선 눈에 띄는 특징으로는, 매수 건수가 매도 건수에 비해 2배가량 높습니다.   
# 이는 인간 심리의 특성으로 설명이 가능합니다.   
# 인간은 본능적으로 이익은 빨리 실현하려고 하고, 손실은 회피하려고 합니다.   
# 이에 따라서 사놓고 파는 행위를 지연해서 매수가 매도보다 많다고 볼 수 있습니다.    
# 
# 또한 20대의 거래 건수는 매도와 매수 모두 다른 연령층에 비해서 낮습니다.    
# 시장진입 시기가 늦은 사람이 많아 상대적으로 짧은 기간 거래를 한 결과일 수 있으므로 뒤에서 다시 해석하도록 하겠습니다.

# In[16]:


#고객별 거래 빈도 계산(한달간 평균 거래 몇번?)

df = kr.groupby(['cus_id', 'orr_dt']).count().reset_index()
df_1 = df.set_index('orr_dt')
df_2 = df_1.groupby('cus_id').resample('M').sum()
df_3 = df_2.reset_index().set_index('orr_dt')
df_4 = df_3.groupby('cus_id').resample('Y').mean()
df_5 = df_4.reset_index()

import datetime as dt

df_5['year'] = df_5['orr_dt'].dt.year
df_7 = df_5[['cus_id', 'year', 'buysell']]
df_8 = df_7.pivot(index='cus_id',columns='year',values='buysell').reset_index()
df_8['freq_2019']=df_8[2019]
df_8['freq_2020']=df_8[2020]
cus_info = pd.merge(cus_info, df_8, on='cus_id', how='left')
cus_info.fillna(0)

df1 = cus_info.loc[cus_info['freq_2019']==0]
df2 = cus_info.loc[cus_info['freq_2019']>0]
df2['freq'] = (df2['freq_2019']+df2['freq_2020'])/2
df1['freq'] = df1['freq_2020']
cus_info['freq'] = pd.merge(df1,df2,on=['cus_id','freq'],how='outer')['freq']


# In[17]:


df = kr.groupby(['cus_id', 'orr_dt']).count().reset_index()
df_1 = df.set_index('orr_dt')
df_2 = df_1.groupby('cus_id').resample('M').sum()
df_3 = df_2.reset_index().set_index('orr_dt')
df1 = df_3.groupby('cus_id').mean()
cus_info['freq'] = pd.merge(df1,cus_info,on=['cus_id'],how='outer')['buysell']


# In[18]:


#고객별 거래빈도 시각화
df2 = cus_info.loc[cus_info['age_cat'] == '20대'].groupby(by=['grade']).median().reset_index()
df3 = cus_info.loc[cus_info['age_cat'] == '30대'].groupby(by=['grade']).median().reset_index()
df4 = cus_info.loc[cus_info['age_cat'] == '40대'].groupby(by=['grade']).median().reset_index()
df5 = cus_info.loc[cus_info['age_cat'] == '50대'].groupby(by=['grade']).median().reset_index()
df6 = cus_info.loc[cus_info['age_cat'] == '60대'].groupby(by=['grade']).median().reset_index()
df7 = cus_info.loc[cus_info['age_cat'] == '70대 이상'].groupby(by=['grade']).median().reset_index()

trace3 = go.Bar(x=df2.grade, y=df2.freq, name='20대',marker_color=col[0])
trace4 = go.Bar(x=df3.grade, y=df3.freq, name='30대',marker_color=col[1])
trace5 = go.Bar(x=df4.grade, y=df4.freq, name='40대',marker_color=col[2])
trace6 = go.Bar(x=df5.grade, y=df5.freq, name='50대',marker_color=col[3])
trace7 = go.Bar(x=df6.grade, y=df6.freq, name='60대',marker_color=col[4])
trace8 = go.Bar(x=df7.grade, y=df7.freq, name='70대 이상',marker_color=col[5])

fig = go.Figure(data=[trace3, trace4, trace5, trace6, trace7,trace8])
fig.update_layout(title='고객별 거래빈도',font=dict(size=18,))
pyo.iplot(fig)


# 연령대별, 자산규모별 거래 빈도를 시각화 해보았습니다.
# 
# 우선, 자산 규모와 거래 빈도는 크게 상관있어 보이지 않습니다.    
# 또 사람들은 평균적으로 한달에 6건 정도의 거래를 하는 것으로 보입니다.    
# 거래 빈도가 높은 것은 두가지 의미로 해석할 수 있습니다.     
# 
# - 종목의 개수가 많음
# 
#     종목의 개수가 많기 때문에 사고파는 경우의 수가 상대적으로 많다고 해석할 수 있습니다.       
#     밑의 그래프를 보았을 때 종목의 개수가 많은 탑클래스의 빈도수가 적어, 종목 개수와는 크게 상관이 없는 것으로 보입니다.
# 
# 
# - 단타 거래를 많이 함
# 
#     주식을 장기적으로 보유하기 보다는 주가의 흐름에 따라 단기에 사고파는 경우가 많다고 해석할 수 있습니다.       
#     
# 

# In[19]:


#고객별 상품 개수 시각화
df2 = cus_info.loc[cus_info['age_cat'] == '20대'].groupby(by=['grade']).median().reset_index()
df3 = cus_info.loc[cus_info['age_cat'] == '30대'].groupby(by=['grade']).median().reset_index()
df4 = cus_info.loc[cus_info['age_cat'] == '40대'].groupby(by=['grade']).median().reset_index()
df5 = cus_info.loc[cus_info['age_cat'] == '50대'].groupby(by=['grade']).median().reset_index()
df6 = cus_info.loc[cus_info['age_cat'] == '60대'].groupby(by=['grade']).median().reset_index()
df7 = cus_info.loc[cus_info['age_cat'] == '70대 이상'].groupby(by=['grade']).median().reset_index()

trace3 = go.Bar(x=df2.grade, y=df2.iem_num_kr, name='20대',marker_color=col[0])
trace4 = go.Bar(x=df3.grade, y=df3.iem_num_kr, name='30대',marker_color=col[1])
trace5 = go.Bar(x=df4.grade, y=df4.iem_num_kr, name='40대',marker_color=col[2])
trace6 = go.Bar(x=df5.grade, y=df5.iem_num_kr, name='50대',marker_color=col[3])
trace7 = go.Bar(x=df6.grade, y=df6.iem_num_kr, name='60대',marker_color=col[4])
trace8 = go.Bar(x=df7.grade, y=df7.iem_num_kr, name='70대 이상',marker_color=col[5])

fig = go.Figure(data=[trace3, trace4, trace5, trace6, trace7,trace8])
fig.update_layout(title='고객별 상품 개수',font=dict(size=18,))
pyo.iplot(fig)


# In[20]:


import plotly.express as px
df = cus_info.loc[(1<cus_info['freq'])&(cus_info['freq']<2000)]
fig = px.scatter(df, x="iem_num_kr", y="freq", color="grade", color_discrete_sequence=col,
                 category_orders={"grade":['01 탑클래스','02 골드','03 로얄','04 그린','05 블루','09 등급없음']},
                labels={
                      "freq": "거래 빈도", 
                      "iem_num_kr": "종목 개수",
                      "grade": "고객 등급"
                     })
fig.update_layout(title='고객별 상품 개수',font=dict(size=18,))

fig.show()


# In[23]:


cus_info.to_csv('cus_info.csv')
kr.to_csv('kr.csv')


# In[24]:


trdkr.to_csv('trdkr.csv')

