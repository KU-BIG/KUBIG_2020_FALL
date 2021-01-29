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


# In[4]:


act = pd.read_csv('2_act_info.csv')
cus = pd.read_csv('2_cus_info.csv')
iem = pd.read_csv('2_iem_info.csv')
trd_kr =  pd.read_csv('2_trd_kr.csv')
trd_oss =  pd.read_csv('2_trd_oss.csv')
cus = cus.astype({'tco_cus_grd_cd' : 'str'})
trd_kr['orr_dt'] = pd.to_datetime(trd_kr['orr_dt'].astype(str), format='%Y%m%d')
trd_oss['orr_dt'] = pd.to_datetime(trd_oss['orr_dt'].astype(str), format='%Y%m%d')
cus_info = pd.read_csv('cus_info.csv')
kr = pd.read_csv('kr.csv')


# In[9]:


trdkr = pd.read_csv('trdkr.csv')
act_cus = pd.read_csv('act_cus.csv')


# ## 1.4 Trading Data - When?

# ### 1.4.1 Korea Trading Data

# In[10]:


df = kr
df = df.sort_values(by=["act_id", "iem_cd","orr_dt","orr_ord"], ascending=[True, True,True,True]).reset_index()
df = df[['act_id','iem_cd','buysell','price','qty','amount','orr_dt','orr_ord']]

n = df.groupby(by=['act_id','iem_cd'], as_index=False).count()['orr_dt'].tolist()
lowsell_col,highsell_col,lowbuy_col,highbuy_col,act_col,iem_col = [],[],[],[],[],[]
r=0
price = df['price'].tolist()
buysell = df['buysell'].tolist()
act = df['act_id'].tolist()
iem = df['iem_cd'].tolist()

for k in range(311922):
    risk,lowsell,highsell,lowbuy,highbuy = 0,0,0,0,0
    for i in range(n[k]-1):
        if price[i]>price[i+1] and buysell[i+1] == 1:
            lowsell = lowsell + 1
        elif price[i]<=price[i+1] and buysell[i+1] == 1:
            highsell = highsell + 1
        elif price[i]>price[i+1] and buysell[i+1] == 2:
            lowbuy = lowbuy + 1
        elif price[i]<=price[i+1] and buysell[i+1] == 2:
            highbuy = highbuy + 1
    lowsell_col.append(lowsell)
    highsell_col.append(highsell)
    lowbuy_col.append(lowbuy)
    highbuy_col.append(highbuy)
    
trdkr = trdkr.sort_values(by=["act_id", "iem_cd"], ascending=[True, True])
data = {'act_id': trdkr['act_id'], 'iem_cd': trdkr['iem_cd'], 'lowsell_kr': lowsell_col,
       'highsell_kr': highsell_col, 'lowbuy_kr': lowbuy_col, 'highbuy_kr': highbuy_col}
data = pd.DataFrame(data)
trdkr = pd.merge(trdkr, data, on=['act_id','iem_cd'])

trdkr['a'] = trdkr['total_num_kr']
trdkr['a'] = trdkr['a'].replace({1:2})
trdkr['lowsell_kr'] = trdkr['lowsell_kr']/(trdkr['a']-1)
trdkr['highsell_kr'] = trdkr['highsell_kr']/(trdkr['a']-1)
trdkr['lowbuy_kr'] = trdkr['lowbuy_kr']/(trdkr['a']-1)
trdkr['highbuy_kr'] = trdkr['highbuy_kr']/(trdkr['a']-1)
trdkr = trdkr.drop('a',axis=1)


# 거래 성향을 파악하기 위해 직전 거래에 비해 가격이 올랐는지 내렸는지 여부와 함께 매도, 매수 여부를 구합니다.     
# 각각의 비중을 구해 다음과 같은 네가지 변수를 추가해주었습니다.
# 
# + lowsell_kr: 주가 떨어졌을 때 매도
# + highsell_kr: 주가 올랐을 때 매도
# + lowbuy_kr: 주가 떨어졌을 때 매수
# + highbuy_kr: 주가 올랐을 때 매수

# In[11]:


#고객별 평균값 구해주기
a = ['lowsell_kr','highsell_kr','lowbuy_kr','highbuy_kr']
data = trdkr.groupby(pd.merge(act_cus, trdkr, on=['act_id'])['cus_id']).mean()[a]
cus_info = pd.merge(data, cus_info, on=['cus_id'],how = 'outer')

cus_info[a]=cus_info[a]*100


# In[12]:


a = ['lowsell_kr','highsell_kr','lowbuy_kr','highbuy_kr']
list(cus_info.mean()[a])


# In[13]:


#연령별 매도/매수 성향 비율 시각화

pie1 = (
    Pie()
    .add("", [list(z) for z in zip(list(cus_info.mean()[a].index), list(cus_info.mean()[a]))], radius=[60, 120])
    .set_global_opts(title_opts=opts.TitleOpts(title="매도/매수 성향 비율 시각화"),
                    legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='20%', pos_top ='middle'))
    .set_series_opts(label_opts=opts.LabelOpts(color=["black"], formatter="{d}%", font_size = 18))
    .set_colors(['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c'])
)

pie1.render_notebook()


# 매도/매수 성향 비율을 시각화 해보았습니다.
# 
# 따로 자산규모별, 연령별, 투자성향별 비율을 시각화 해보았지만, 차이가 1% 내외로 매우 작았습니다.    
# 투자자들의 거래 행동을 살펴보면, 다음과 같이 정리할 수 있습니다.
# 
# + 추격매수
#     
#   가격이 내려갈 때 저점이라고 생각해서 매수: 거래 행동의 대다수(65% 이상)을 차지
#     
#   가격이 올라갈 때 상승할 것이라고 생각해서 매수: 거래 행동의 20% 이상을 차지
#     
#     
# + 손절매
# 
#   가격이 내려갈 때 손해를 방지하기 위해 매도: 거래행동의 약 0.7% 차지
#   
#   
#   
# 
# + 익절매
# 
#   가격이 올라갈 때 차익을 얻기 위해 매도: 거래행동의 약 2% 차지
#     
#     
# 이러한 양상은 앞서 언급했었던 손해를 미루고 이익을 추구하는 인간의 기본 심리와도 관련이 있습니다.     
# 또한 매수에 있어서도 오르고 있는 주식을 사기 보다는, 가격이 내려가는 주식을 사려고 합니다.      
# 이는 투자성향이나 자산규모, 연령과 상관없이 비율이 비슷하게 나옵니다.

# In[18]:


cus_info = cus_info.fillna(0)

#계좌-상품별 순매수금액 구하기
trdkr['net_purchase_kr']=trdkr['buy_amount_kr']-trdkr['sell_amount_kr']

#고객별 순매수금액 구하기
cus_info['net_purchase_kr']=cus_info['buy_amount_kr']-cus_info['sell_amount_kr']

#필요한 coulmn만 추출
cus_info = cus_info[['cus_id','age_cat','sex','grade','class','risk', 'ivst', 'home', 'num_act', 'iem_num_kr',
                     'lowsell_kr', 'highsell_kr', 'lowbuy_kr', 'highbuy_kr',
                     'sell_qty_kr', 'sell_amount_kr', 'buy_qty_kr', 'buy_amount_kr',
                     'sell_num_kr', 'buy_num_kr', 'total_num_kr','freq_2019','freq_2020','freq']]


# 여러 주식투자 분석 레포트에서 '순매수금액 및 비중'을 중요한 지표로 활용하는 것을 보았습니다.    
# 그래서 순매수 금액을 추출해 **net_purchase_kr**과 **net_purchase_oss** 변수를 추가했습니다.

# In[20]:


cus_info.to_csv('cus_info.csv')
trdkr.to_csv('trdkr.csv')

