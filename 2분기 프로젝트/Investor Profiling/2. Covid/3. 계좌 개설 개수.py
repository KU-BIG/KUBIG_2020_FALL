#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


act = pd.read_csv('2_act_info.csv')
cus = pd.read_csv('2_cus_info.csv')
iem = pd.read_csv('2_iem_info.csv')
trd_kr =  pd.read_csv('2_trd_kr.csv')
trd_oss =  pd.read_csv('2_trd_oss.csv')
cus = cus.astype({'tco_cus_grd_cd' : 'str'})
trd_kr['orr_dt'] = pd.to_datetime(trd_kr['orr_dt'].astype(str), format='%Y%m%d')
trd_oss['orr_dt'] = pd.to_datetime(trd_oss['orr_dt'].astype(str), format='%Y%m%d')


# In[3]:


#연령대 카테고리 추가

cus.loc[(cus['cus_age'] >= 0) & (cus['cus_age'] < 10), 'age_cat'] = '10대 미만'
cus.loc[(cus['cus_age'] >= 10) & (cus['cus_age'] < 20), 'age_cat'] = '10대'
cus.loc[(cus['cus_age'] >= 20) & (cus['cus_age'] < 30), 'age_cat'] = '20대'
cus.loc[(cus['cus_age'] >= 30) & (cus['cus_age'] < 40), 'age_cat'] = '30대'
cus.loc[(cus['cus_age'] >= 40) & (cus['cus_age'] < 50),'age_cat'] = '40대'
cus.loc[(cus['cus_age'] >= 50) & (cus['cus_age'] < 60), 'age_cat'] = '50대'
cus.loc[(cus['cus_age'] >= 60) & (cus['cus_age'] < 70), 'age_cat'] = '60대'
cus.loc[cus['cus_age'] == 70, 'age_cat'] = '70대 이상'

#고객 등급, 투자성향 변수 변경

cus['tco_cus_grd_cd'] = cus['tco_cus_grd_cd'].replace({'01':'01 탑클래스','02':'02 골드','03':'03 로얄','04':'04 그린',
                                                       '05':'05 블루','09':'09 등급없음','_ ':'09 등급없음'})
cus['ivs_icn_cd'] = cus['ivs_icn_cd'].replace({'01':'01 안정형','02':'02 안정추구형','03':'03 위험중립형','04':'04 적극투자형',
                                               '05':'05 공격투자형','09':'09 전문투자자형','00':'정보 제공 미동의',
                                               '_ ':'등급없음', '_':'등급없음','-':'등급없음'})
cus = cus.rename({"sex_dit_cd":"sex", "cus_age":"age", "zip_ctp_cd": "home",
                  "tco_cus_grd_cd":"grade", "ivs_icn_cd":"ivst"},axis='columns')


# In[4]:


actcus = pd.merge(act,cus,on='cus_id')
act_cus = actcus[['act_id','cus_id','act_opn_ym']]
act_cus.to_csv('act_cus.csv', index=False) #고객정보와 계좌정보 연결한 데이터


# In[5]:


#개인당 국내계좌 개수 추출

count = {}
lists = list(actcus['cus_id'])

for i in lists:
    try: count[i] += 1
    except: count[i]=1
        
a = pd.DataFrame({'cus_id' : list(count.keys()), 
                  'num_act' : list(count.values())})


# In[6]:


cus_info = pd.merge(cus, a, on='cus_id') #고객정보 데이터(10000명)


# In[7]:


cus_info1 = pd.merge(cus_info, act, on='cus_id', how='left')
cus_info1


# In[8]:


cus_info1['act_opn_ym'] = cus_info1['act_opn_ym'].astype('str')
cus_info1['opn_ym'] = cus_info1['act_opn_ym'].str[:4]
cus_info1['opn_ym'] = cus_info1['opn_ym'].astype('int')


# In[9]:


cus_info1


# In[10]:


cond = cus_info1['opn_ym'] < 2018
cus_info1['opn_y'] = np.where(cond, '2018년_이전', cus_info1['opn_ym'])


# In[11]:


df = cus_info1.groupby(['opn_y', 'age_cat']).count().reset_index()


# In[12]:


df = df[['opn_y', 'age_cat', 'cus_id']]
df


# In[13]:


df1 = df.pivot(index='age_cat',columns='opn_y',values='cus_id').reset_index()
df1 = df1.rename(columns={'2018': "year_2018", '2019': "year_2019", '2020': "year_2020"})


# In[14]:


df1


# In[15]:


trace1 = go.Bar(x=df1.age_cat, y=df1.year_2018, name='2018년',marker_color=col[2])
trace2 = go.Bar(x=df1.age_cat, y=df1.year_2019, name='2019년',marker_color=col[1])
trace3 = go.Bar(x=df1.age_cat, y=df1.year_2020, name='2020년',marker_color=col[0])

fig = go.Figure(data=[trace1, trace2, trace3], layout=go.Layout(title='연도별 계좌 개설 개수'))
pyo.iplot(fig)


# In[16]:


#2020년 10월까지만 데이터 존재
cus_info1[cus_info1.opn_ym == 2020].act_opn_ym.unique()


# In[17]:


#일까지 정보가 없어서 에러가 나는 듯,,,
#cus_info1['act_opn_ym'] = cus_info1['act_opn_ym'].astype('str')
#pd.datetime(cus_info1['act_opn_ym'], format='%Y%m')


# In[18]:


cus_info1['act_opn_ym'].unique()


# In[41]:


cus_info1['act_opn_ym'] = cus_info1['act_opn_ym'].astype('int')
df_1920 = cus_info1[cus_info1['act_opn_ym'] > 201900]
df_20 = df_1920[(df_1920['age_cat'] == '20대') | (df_1920['age_cat'] == '30대')]
df2 = df_20.groupby(['act_opn_ym', 'age_cat']).count().reset_index()[['act_opn_ym', 'age_cat', 'cus_id']]


# In[42]:


df3 = df2.pivot(index='act_opn_ym',columns='age_cat',values='cus_id').reset_index()
df3['act_opn'] = df3.act_opn_ym.astype('str').str[:4] + '-' + df3.act_opn_ym.astype('str').str[4:] 


# In[43]:


df3


# In[44]:


trace3 = go.Bar(x=df3.act_opn, y=df3['20대'], name='20대',marker_color=col[5])
trace4 = go.Bar(x=df3.act_opn, y=df3['30대'], name='30대',marker_color=col[6])


data = [trace3, trace4]

layout = go.Layout(title='월별 계좌개설 개수(2019년-2020년)', barmode='stack')

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# In[45]:


trace3 = go.Bar(x=df3.act_opn, y=df3['20대'], name='20대',marker_color='#F3BE50')
trace4 = go.Bar(x=df3.act_opn, y=df3['30대'], name='30대',marker_color='#2C66B5')


data = [trace3, trace4]

layout = go.Layout(title='월별 계좌개설 개수(2019년-2020년)')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# In[24]:


df_181920 = cus_info1[cus_info1['act_opn_ym'] > 201800]
df_2030 = df_181920[(df_181920['age_cat'] == '20대') | (df_181920['age_cat'] == '30대')]
df4 = df_2030.groupby(['act_opn_ym', 'age_cat']).count().reset_index()[['act_opn_ym', 'age_cat', 'cus_id']]


# In[25]:


df4 = df4.pivot(index='act_opn_ym',columns='age_cat',values='cus_id').reset_index()
df4['act_opn'] = df4.act_opn_ym.astype('str').str[:4] + '-' + df4.act_opn_ym.astype('str').str[4:] 


# In[26]:


trace3 = go.Bar(x=df4.act_opn, y=df4['20대'], name='20대',marker_color=col[5])
trace4 = go.Bar(x=df4.act_opn, y=df4['30대'], name='30대',marker_color=col[6])


data = [trace3, trace4]

layout = go.Layout(title='월별 계좌개설 개수(2018년-2020년)', barmode='stack')

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# In[27]:


trace3 = go.Bar(x=df4.act_opn, y=df4['20대'], name='20대',marker_color=col[4])
trace4 = go.Bar(x=df4.act_opn, y=df4['30대'], name='30대',marker_color=col[2])


data = [trace3, trace4]

layout = go.Layout(title='월별 계좌개설 개수(2018년-2020년)')

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# In[33]:


act = pd.read_csv('2_act_info.csv')
cus = pd.read_csv('2_cus_info.csv')
iem = pd.read_csv('2_iem_info.csv')
trd_kr =  pd.read_csv('2_trd_kr.csv')
trd_oss =  pd.read_csv('2_trd_oss.csv')
cus = cus.astype({'tco_cus_grd_cd' : 'str'})
trd_kr['orr_dt'] = pd.to_datetime(trd_kr['orr_dt'].astype(str), format='%Y%m%d')
trd_oss['orr_dt'] = pd.to_datetime(trd_oss['orr_dt'].astype(str), format='%Y%m%d')

trd_kr = pd.merge(act_cus, trd_kr, on='act_id') #거래-고객 정보 데이터
trd_kr['amount'] = trd_kr['cns_qty']*trd_kr['orr_pr'] #거래별 금액 계산
kr = trd_kr.rename({'orr_rtn_hur':'ord','lst_cns_hur':'real','sby_dit_cd':'buysell', 
                        'orr_mdi_dit_cd':'store', 'cns_qty':'qty', 'orr_pr':'price'},axis='columns')


# In[34]:


cus_info = pd.read_csv('cus_info.csv')


# In[35]:


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


# In[32]:


df


# In[46]:


df2 = df.loc[(df['age_cat'] == '20대')&(df['covid'] == 'after')]
df_date = df2[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_20 = df_date.resample('W').mean().reset_index()

df3 = df.loc[(df['age_cat'] == '30대')&(df['covid'] == 'after')]
df_date = df3[['orr_dt','haha']]
df_date = df_date.groupby(['orr_dt']).sum()
df_30 = df_date.resample('W').mean().reset_index()


trace3 = go.Bar(x=df_20.orr_dt, y=df_20.haha, name='20대',marker_color='#F3BE50')
trace4 = go.Bar(x=df_30.orr_dt, y=df_30.haha, name='30대',marker_color='#2C66B5')

data = [trace3, trace4]

layout = go.Layout(title='30대 거래건수 흐름')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# In[37]:


data = [trace4]

layout = go.Layout(title='30대 거래건수 흐름', barmode='stack')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(font=dict(size=15,))
pyo.iplot(fig)


# In[49]:


cus_info = pd.read_csv('cus_info2.csv')


# In[119]:


#고객별 종목 산업 개수 시각화

df2 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.ind_cat, name='20대 남성',marker_color='#2C66B5')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.ind_cat, name='20대 여성',marker_color='#F3BE50')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.ind_cat, name='30대 남성',marker_color='#2C66B5')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.ind_cat, name='30대 여성',marker_color='#F3BE50')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 산업 개수'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)


# In[120]:


df2 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.iem_num_kr, name='20대 남성',marker_color='#2C66B5')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.iem_num_kr, name='20대 여성',marker_color='#F3BE50')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.iem_num_kr, name='30대 남성',marker_color='#2C66B5')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.iem_num_kr, name='30대 여성',marker_color='#F3BE50')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 상품 개수'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)


# In[55]:


cus_info.loc[cus_info['freq'] == 0]


# In[142]:


df2 = cus_inf.loc[(cus_inf['age_cat']=='20대') & (cus_inf['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_inf.loc[(cus_inf['age_cat']=='20대') & (cus_inf['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_inf.loc[(cus_inf['age_cat']=='30대') & (cus_inf['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_inf.loc[(cus_inf['age_cat']=='30대') & (cus_inf['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.freq, name='20대 남성',marker_color='#2C66B5')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.freq, name='20대 여성',marker_color='#F3BE50')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.freq, name='30대 남성',marker_color='#2C66B5')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.freq, name='30대 여성',marker_color='#F3BE50')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 거래 빈도'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)

