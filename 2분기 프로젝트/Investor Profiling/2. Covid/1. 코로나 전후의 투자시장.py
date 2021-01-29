#!/usr/bin/env python
# coding: utf-8

# In[51]:


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


# In[52]:


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
trdkr = pd.read_csv('trdkr.csv')
kr = pd.read_csv('kr.csv')
oss = pd.read_csv('oss.csv')
kr['orr_dt'] = pd.to_datetime(kr['orr_dt'].astype(str), format='%Y-%m-%d')
oss['orr_dt'] = pd.to_datetime(oss['orr_dt'].astype(str), format='%Y-%m-%d')


# In[53]:


item_n = pd.read_csv('item_n.csv')


# In[54]:


oss = oss.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)


# In[55]:


kr = kr.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)


# ## 3.1 시간 흐름에 따른 데이터 분석

# In[56]:


#코로나 이전 이후 그룹 나누기

kr2=kr[['cus_id','orr_dt']]
oss2=oss[['cus_id','orr_dt']]
df = pd.merge(kr2,oss2,on=['cus_id','orr_dt'],how='outer')
df = df.groupby(by=['cus_id']).min()

from datetime import datetime
df1 = df.loc[df['orr_dt']<datetime(2020, 1, 1)].reset_index()
df2 = df.loc[df['orr_dt']>datetime(2020, 1, 1)].reset_index()

df1['covid']='before'
df2['covid']='after'

df1 = df1[['cus_id','covid']]
df2 = df2[['cus_id','covid']]

df = pd.merge(df1,df2,on=['cus_id','covid'],how='outer')
cus_info = pd.merge(cus_info,df,on=['cus_id'],how='outer')


# 코로나 전에 시장에 있던 사람과 코로나 이후에 진입한 사람을 나누었습니다. 
# 분류한 뒤 **covid**라는 변수도 함께 추가해주었습니다.    
# 분류 결과 각 그룹에 속하는 사람의 수는 다음과 같습니다.
# 
# + 코로나 이전: 6046명
# + 코로나 이후: 3954명

# In[57]:


df = kr
df_date = df[['orr_dt', 'buysell', 'amount']]
df_date = df_date.groupby(['orr_dt', 'buysell']).sum()

net_purchase = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        a = df_date['amount'][i+1] - df_date['amount'][i]
        net_purchase.append(a)
        
date = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        date.append(df_date.index[i][0])
        
date_buy = pd.DataFrame({'date' : date, 'net_purchase' : net_purchase})
date_buy = date_buy.set_index('date') 
dff = date_buy.resample('W').mean().reset_index()
df = pd.merge(df, cus_info, on='cus_id')


# In[58]:


df_date


# 우선 데이터를 통해서 일주일별로 주식 순매수 금액이 어떻게 되는지 구해보았습니다.

# In[59]:


df2 = df.loc[df['age_cat'] == '20대']
df_date = df2[['orr_dt', 'buysell', 'amount']]
df_date = df_date.groupby(['orr_dt', 'buysell']).sum()

net_purchase = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        a = df_date['amount'][i+1] - df_date['amount'][i]
        net_purchase.append(a)
        
date = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        date.append(df_date.index[i][0])
        
date_buy = pd.DataFrame({'date' : date, 'net_purchase' : net_purchase})
date_buy = date_buy.set_index('date') 
df_20 = date_buy.resample('W').mean().reset_index()


# In[60]:


df3 = df.loc[df['age_cat'] == '30대']
df_date = df3[['orr_dt', 'buysell', 'amount']]
df_date = df_date.groupby(['orr_dt', 'buysell']).sum()

net_purchase = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        a = df_date['amount'][i+1] - df_date['amount'][i]
        net_purchase.append(a)
        
date = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        date.append(df_date.index[i][0])
        
date_buy = pd.DataFrame({'date' : date, 'net_purchase' : net_purchase})
date_buy = date_buy.set_index('date') 
df_30 = date_buy.resample('W').mean().reset_index()


# 저희의 분석대상인 2,30대의 경우도 일주일별 주식 순매수 금액을 구해보았습니다.

# In[61]:


df4 = df.loc[(df['age_cat'] != '20대') & (df['age_cat'] != '30대')]
df_date = df4[['orr_dt', 'buysell', 'amount']]
df_date = df_date.groupby(['orr_dt', 'buysell']).sum()

net_purchase = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        a = df_date['amount'][i+1] - df_date['amount'][i]
        net_purchase.append(a)
        
date = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        date.append(df_date.index[i][0])
        
date_buy = pd.DataFrame({'date' : date, 'net_purchase' : net_purchase})
date_buy = date_buy.set_index('date') 
df_2030 = date_buy.resample('W').mean().reset_index()


# In[62]:


df4 = df.loc[(df['age_cat'] != '20대') & (df['age_cat'] != '30대')]
df_date = df4[['orr_dt', 'buysell', 'amount']]
df_date = df_date.groupby(['orr_dt', 'buysell']).sum()

net_purchase = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        a = df_date['amount'][i+1] - df_date['amount'][i]
        net_purchase.append(a)
        
date = []

for i in range(0, len(df_date['amount'])):
    if i % 2 == 0:
        date.append(df_date.index[i][0])
        
date_buy = pd.DataFrame({'date' : date, 'net_purchase' : net_purchase})
date_buy = date_buy.set_index('date') 
df_2030 = date_buy.resample('W').mean().reset_index()


# In[63]:


df


# In[64]:


df_date = df.loc[(df['age_cat'] == '20대')&(df['covid'] == 'after')][['orr_dt','buysell','qty','amount','price']]


# In[65]:


df_date = df_date.groupby(['orr_dt', 'buysell']).sum()
df_date.index[2]


# 그래프 구현을 위해 2,30대 외의의 경우도 일주일별 주식 순매수 금액을 구해보았습니다.

# In[66]:


#순매수 금액 흐름

trace3 = go.Bar(x=df_20.date, y=df_20.net_purchase, name='20대',marker_color='#F3BE50')
trace4 = go.Bar(x=df_30.date, y=df_30.net_purchase, name='30대',marker_color='#2C66B5')
trace5 = go.Bar(x=df_2030.date, y=df_2030.net_purchase, name='그 외',marker_color=col[10])

data = [trace3, trace4, trace5]

layout = go.Layout(title='순매수 금액 흐름', barmode='stack')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# In[67]:


df = kr
df['haha'] = 1
df_date = df[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
dff = df_date.resample('W').mean().reset_index()

df = pd.merge(df, cus_info, on='cus_id')

df2 = df.loc[df['age_cat'] == '20대']
df_date = df2[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_20 = df_date.resample('W').mean().reset_index()

df3 = df.loc[df['age_cat'] == '30대']
df_date = df3[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_30 = df_date.resample('W').mean().reset_index()

df4 = df.loc[(df['age_cat'] != '20대') & (df['age_cat'] != '30대')]
df_date = df4[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_2030 = df_date.resample('W').mean().reset_index()


# In[68]:


df2 = df.loc[(df['age_cat'] == '30대')&(df['covid'] == 'before')]
df_date = df2[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_20 = df_date.resample('W').mean().reset_index()

df3 = df.loc[(df['age_cat'] == '20대')&(df['covid'] == 'after')]
df_date = df3[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_30 = df_date.resample('W').mean().reset_index()


trace3 = go.Bar(x=df_20.orr_dt, y=df_20.haha, name='코로나 이전',marker_color='#F3BE50')
trace4 = go.Bar(x=df_30.orr_dt, y=df_30.haha, name='코로나 이후',marker_color='#2C66B5')

data = [trace3, trace4]

layout = go.Layout(title='30대 거래건수 흐름', barmode='stack')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# In[69]:


data = [trace4]

layout = go.Layout(title='30대 거래건수 흐름', barmode='stack')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# 일주일별 순매수 금액 흐름을 시각화 해보았습니다
# 
# 2020년 1월을 기점으로 순매수 금액이 폭발적으로 증가함을 볼 수 있습니다.
# 이 때 최근 10년간 주가의 최저치를 찍었었습니다.   
# 소위 '개미'라고 불리우는, 주가가 떨어지면서 하락장에 참여한 매수자들이 매우 많았던 시기입니다.     
# 이 때 늘어난 순매수금액에 2,30대의 비중도 늘어나는 것처럼 보이는데, 더 확실하게 알아보기 위해 그래프를 그려보았습니다.

# In[70]:


#순매수 금액 흐름

trace3 = go.Bar(x=df_20.orr_dt, y=df_20.haha, name='20대',marker_color=col[4])
trace4 = go.Bar(x=df_30.orr_dt, y=df_30.haha, name='30대',marker_color=col[3])
trace5 = go.Bar(x=df_2030.orr_dt, y=df_2030.haha, name='그 외',marker_color=col[10])

data = [trace3, trace4, trace5]

layout = go.Layout(title='거래 건수 흐름', barmode='stack')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# 30대의 경우 코로나19를 기점으로 순매수 금액이 늘어났다고 보기는 어렵지만, 20대의 경우 차지하는 비율이 2배 이상 증가함을 볼 수 있습니다.    
# 코로나19 이후 주식시장에 유입된 인구중 상당수가 20대이며, 이들의 매수량이 많이 늘어났다고 할 수 있습니다.

# In[71]:


final_data = pd.merge(kr, item_n, on='iem_cd')
final_data[final_data['ind_cat'] == '기타']
df_c = pd.DataFrame(final_data.groupby(['orr_dt', 'ind_cat'])['buysell'].count())
df_c = df_c.unstack().resample('W').mean().sort_index()

a1 = list(df_c['buysell']['에너지'])
b1 = list(df_c['buysell']['소재'])
c1 = list(df_c['buysell']['산업재'])
d1 = list(df_c['buysell']['경기소비재'])
e1 = list(df_c['buysell']['필수소비재'])
f1 = list(df_c['buysell']['의료'])
g1 = list(df_c['buysell']['금융'])
h1 = list(df_c['buysell']['IT'])
i1 = list(df_c['buysell']['통신서비스'])
j1 = list(df_c['buysell']['기타'])

date = df_c['buysell']['에너지'].index.tolist() 


# In[72]:


df_c.unstack()


# In[73]:


#날짜별 산업 거래 건수 시각화

line_chart = (
        Line()
        .add_xaxis(date)
        .add_yaxis("에너지", a1, color='skyblue')
        .add_yaxis("소재", b1, color='orange')
        .add_yaxis("산업재", c1, color='pink')
        .add_yaxis("경기소비재", d1, color='red')
        .add_yaxis("필수소비재", e1, color='black')
        .add_yaxis("의료", f1, color='blue')
        .add_yaxis("금융", g1, color='green')
        .add_yaxis("IT", h1, color='purple')
        .add_yaxis("통신서비스", i1, color='darkblue')
        .add_yaxis("기타", i1, color='gray')
    
        .set_global_opts(title_opts=opts.TitleOpts(title="산업별 거래"),
                         xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),
                         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")])
        .set_series_opts(label_opts=opts.LabelOpts(is_show = False,position="top", color=["black"]))
)
line_chart.render_notebook()


# 산업별 종목의 일별 거래 횟수를 시각화하여 그 거래 흐름을 살펴보았습니다.   
# 
# 전반적으로 주식 거래가 <span style="color:blue"> 2020년을 분기점으로 활발해짐</span>을 확인할 수 있습니다.       
# 특히, 의료 부문의 거래 빈도가 2020년 이후 가파르게 증가하였는데 이는 코로나19 상황과 관련이 있다고 보입니다.      
# 또한, 의료 부문 거래가 가장 활발하게 이루어진 2020년 3월은 코로나19가 대유행하던 시기와 맞물립니다.

# In[74]:


cus_info.to_csv('cus_info.csv')

