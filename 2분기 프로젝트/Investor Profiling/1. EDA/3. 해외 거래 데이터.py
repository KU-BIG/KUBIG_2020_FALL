#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


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


# ## 1.3 Overseas Trading Data

# In[5]:


trd_oss = pd.merge(act_cus, trd_oss, on='act_id') #거래-고객 정보 데이터
trd_oss['amount'] = trd_oss['cns_qty']*trd_oss['orr_pr']*trd_oss['trd_cur_xcg_rt'] #거래별 금액 계산


# In[6]:


#고객별 종목 개수

a2 = dict(list(trd_oss[['cus_id','iem_cd']].groupby('cus_id')))
b2 = trd_oss.cus_id.unique().tolist()
e2 = []

for i in b2:
  c2 = a2[i]
  d2 = len(c2.iem_cd.unique())
  e2.append(d2)

df = pd.DataFrame({'cus_id': b2, 'iem_num_oss': e2})
cus_info = pd.merge(cus_info, df, on='cus_id',how='outer')


# In[7]:


oss = trd_oss.rename({'orr_rtn_hur':'ord','lst_cns_hur':'real','sby_dit_cd':'buysell', 
                      'orr_mdi_dit_cd':'store', 'cns_qty':'qty', 'orr_pr':'price',
                      'cur_cd':'country','trd_cur_xcg_rt':'rate'},axis='columns')


# In[8]:


df = oss

#매도, 매수 데이터 분할
df_sell = df.loc[df['buysell']==1]
df_buy = df.loc[df['buysell']==2]
df_sell = df_sell.groupby(by=['act_id','iem_cd'], as_index=False)
df_buy = df_buy.groupby(by=['act_id','iem_cd'], as_index=False)

#고객, 종목별 매도/매수 주식개수 총합, 거래액 총합 계산
act_sell = df_sell.sum()[['act_id','iem_cd','qty','amount']].fillna(0)
act_buy = df_buy.sum()[['act_id','iem_cd','qty','amount']].fillna(0)

act_sell.rename(columns={'amount' : 'sell_amount_oss', 'qty':'sell_qty_oss'}, inplace=True)
act_buy.rename(columns={'amount' : 'buy_amount_oss', 'qty':'buy_qty_oss'}, inplace=True)

#고객, 종목별 매도/매수 건수 계산
count_sell = df_sell.count()[['act_id','iem_cd','buysell']].fillna(0)
count_buy = df_buy.count()[['act_id','iem_cd','buysell']].fillna(0)
count_sell.rename(columns={'buysell' : 'sell_num_oss'}, inplace=True)
count_buy.rename(columns={'buysell' : 'buy_num_oss'}, inplace=True)

#구한 값들을 합쳐줍니다.
df = pd.merge(act_sell, act_buy, on=['act_id','iem_cd'],how="outer")
df = pd.merge(df, count_sell, on=['act_id','iem_cd'],how="outer")
df = pd.merge(df, count_buy, on=['act_id','iem_cd'],how="outer")
trdoss = df.fillna(0) #해외: 계좌 - 종목별 매도/매수 건수, 총합 들어있는 데이터
trdoss['total_num_oss'] = trdoss['buy_num_oss'] + trdoss['sell_num_oss']


# In[9]:


#계좌별 매도/매수 건수 계산
trdoss2 = trdoss.groupby(trdoss['act_id']).sum()

#고객별 매도/매수 건수 계산
trdoss3 = pd.merge(trdoss2, act_cus, on=['act_id'])
trdoss4 = trdoss3.groupby(trdoss3['cus_id']).sum()
cus_info = pd.merge(trdoss4, cus_info, on=['cus_id'], how='outer')


# 종목별 해외 매도, 매수 주식 개수를 추출해 **sell_qty_oss**, **buy_qty_oss** 변수를 만듭니다.    
# 종목별 해외 매도, 매수 주식 액수를 추출해 **sell_amount_oss**, **buy_amount_oss** 변수를 만듭니다.    
# 종목별 해외 매도, 매수 주식거래 건수를 추출해 **sell_num_oss**, **buy_num_oss** 변수를 만들고, 둘을 합쳐 **total_num_oss** 변수를 만듭니다.    

# In[10]:


#해외 거래 건수 비율 시각화(해외거래를 한 사람에 한해서)

df = cus_info
df['ratio'] = 100*df['total_num_oss']/(df['total_num_oss']+df['total_num_kr'])
df = df.loc[df['total_num_oss'] > 0]

fig = px.box(df, x="age_cat", y="ratio", color="age_cat",
                category_orders={"age_cat": ["10대 미만","20대","30대","40대","50대","60대","70대 이상"]},
                labels={
                        "age_cat": "연령대",
                        "ratio": "비율(%)",
                       })
fig.update_layout(title='고객별 전체 거래 대비 해외 거래 건수')
pyo.iplot(fig)


# 전체 거래 대비 해외 거래 건수를 살펴보았을 때, 30대의 비율이 비교적 낮은 편임을 확인할 수 있었습니다.   
# 즉, 30대는 국내 거래를 위주로 하고, 20대는 30대에 비해 상대적으로 해외 투자를 많이 합니다.
# 
# 또한 해외투자는 조금 더 신세대인 2,30대가 할 법하다고 생각할 수 있는데, 이와 달리 10대 미만과 60대에서 해외투자의 비율이 매우 높습니다.    
# 이는 시장의 해외투자가 개인이 아닌 PB(Private Banker)에 의해 이루어지고 있음을 의심해볼 수도 있습니다.    
# 그래서 상대적으로 자산규모가 낮고, 개인 투자를 하는 2,30대의 해외 투자 비율이 낮아진다고 생각할 수도 있습니다.    

# In[11]:


#국내, 해외 거래 상품개수 시각화

df1 = cus_info.loc[cus_info['total_num_oss'] > 0]
df1 = df1.groupby(by=['age_cat']).median().reset_index()
df = cus_info.groupby(by=['age_cat']).median().reset_index()

trace3 = go.Bar(x=df.age_cat, y=df.iem_num_kr, name='국내 상품',marker_color=col[0])
trace4 = go.Bar(x=df1.age_cat, y=df1.iem_num_oss, name='해외 상품',marker_color=col[3])

data = [trace3, trace4]

layout = go.Layout(title='고객별 상품 개수(중앙값)')

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# 보유하고 있는 종목 개수에 있어서도 해외 상품의 개수가 국내 상품보다 현저히 적습니다.      
# 또한 연령대별로도 거의 차이가 나지 않습니다.    
# 이는 우리나라에서 해외주식 직접투자가 가능해진지 얼마 되지 않았기 때문일 확률이 높습니다.
# 
# 사람들은 평균적으로 10개 정도 종목에 분산투자를 하고 있으며, 30대의 경우 12개로 조금 더 높습니다.

# In[13]:


cus_info.to_csv('cus_info.csv')
oss.to_csv('oss.csv')
trdoss.to_csv('trdoss.csv')

