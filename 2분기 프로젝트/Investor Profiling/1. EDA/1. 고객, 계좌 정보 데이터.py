#!/usr/bin/env python
# coding: utf-8

# # Post Corona: From Crisis to Opportunity
# ## KUBIG 127
# 
# ### 2020 12 31
# 
# #### 팀명 : 쿠빅 127
# 
# #### 코로나 전후 Y&Z 세대의 투자행동 비교분석 
# 
# 
# 
# 2020년 1월 등장한 코로나는 우리의 생활을 송두리째 뒤바꿔 놓았습니다.      
# 마스크 없이는 밖을 나가지 못하게 되었으며, 항공업, 관광업, 예술산업 등 수많은 산업에 경제적인 타격을 주었습니다.      
# 하지만 여기 코로나 통해서 위기를 통해 기회를 잡은 사람들이 있습니다.            
# 밖에서 소비를 하지 않고 방구석에 계속 있어서인지(!) 정부정책으로 시장의 유동성이 커져서인지,       
# 주식시장에 처음으로 진입해 수익을 봤다는 2,30대 친구들의 소식이 이곳저곳에서 쏠쏠하게 들려옵니다.         
# 그래서 저희는 코로나 전 후의 시장 상황이 어떤지, 또 코로나 이후 진입한 투자자들의 특징은 무엇인지 분석해 보기로 했습니다.      
# 
# -----------------------
# [코드 순서]
#     
#     
# 0. Import 
#    
#    
# 1. EDA     
#     1.1 고객 및 계좌 데이터        
#     1.2 국내 거래 데이터       
#     1.3 해외 거래 데이터      
#     1.4 매도/매수 데이터     
#     1.5 주식 상품 데이터      
#   
# 
#     
# 2. Modeling    
#     2.1 고객 투자성향 예측모델    
#     2.2 투자성향에 중요한 변수? 
#     
#     
#     
# 3. Influence of COVID19    
#     3.1 시간 흐름에 따른 분석        
#     3.2 코로나 전후 그룹 특성 비교분석  
#     3.3 집단별 특성 비교분석
#     
#    
#  
# 4. Insight  
# 
# 
# 5. Appendix     
#     5.1 가설 검증      
#     5.2 참고 자료

# # 0. Import

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


# # 1. EDA

# ## 1.1 고객 및 계좌 데이터

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


#고객별 등급 분포 시각화

df = cus_info.groupby(by=['age_cat', 'grade']).count()
df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()

fig = px.bar(df, x='age_cat', y='cus_id', color='grade',color_discrete_sequence=px.colors.qualitative.Pastel,
             category_orders={"grade":['01 탑클래스','02 골드','03 로얄','04 그린','05 블루','09 등급없음'],
                             "age_cat": ["10대 미만","20대","30대","40대","50대","60대","70대 이상"]},
             labels={
                      "age_cat": "연령대",
                      "cus_id": "비율(%)",
                      "grade": "고객 등급"
                     },)

fig.update_layout(title='고객별 등급 분포',
                 font=dict(size=18,))

fig.show()


# 우선 연령대별로 자산 규모를 확인해 보았습니다.
# 등급이 없는 경우 자산이 1000만원 이하라고 가정했습니다.
# 
# 저희가 분석 대상으로 하는 2,30대는 자산 규모가 상대적으로 작습니다.    
# 특히 20대의 경우 자산이 3천만원 이하인 사람의 비중이 95%가 넘고, 30대의 경우 90%가 넘습니다.    
# 다시 말해, 2,30대는 **작은 규모**의 자산을 가지고 주식투자를 하고 있습니다.   

# In[8]:


#고객별 투자성향 분포 시각화

df = cus_info.groupby(by=['age_cat', 'ivst']).count()
df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()

fig = px.bar(df, x='age_cat', y='cus_id', color='ivst',
             category_orders={"age_cat": ["10대 미만","20대","30대","40대","50대","60대","70대 이상"],
                             "ivst":['01 안정형','02 안정추구형','03 위험중립형','04 적극투자형','05 공격투자형','09 전문투자자형','등급없음','정보 제공 미동의']},
             labels={
                      "age_cat": "연령대",
                      "cus_id": "비율(%)",
                      "ivst": "고객 투자 성향"
                     },)

fig.update_layout(title='고객별 투자성향 분포',font=dict(size=18,))
fig.show()


# 또 연령대별로 주관적인 투자 성향을 확인해보았습니다.
# 
# 20대의 경우 타 연령층에 비해 안정형의 비율이 매우 높습니다.    
# 또한 20대에서 30대로 넘어가면서, 위험을 감수하는 성향이 많아집니다.

# In[9]:


#고객별 등급 분포 시각화

cus_info['risk'] = cus_info['ivst'].replace({'01 안정형':'01 위험회피형','02 안정추구형':'01 위험회피형','03 위험중립형':'02 위험중립형',
                                             '04 적극투자형':'03 위험감수형', '05 공격투자형':'03 위험감수형'})
df = cus_info.groupby(by=['grade','risk']).count()
df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()

fig = px.bar(df, x='grade', y='cus_id', color='risk',
             category_orders={"grade":['01 탑클래스','02 골드','03 로얄','04 그린','05 블루','09 등급없음'],
                              "risk": ['01 위험회피형','02 위험중립형','03 위험감수형','09 전문투자자형','등급없음','정보 제공 미동의']},
             labels={
                      "risk": "고객 투자 성향", 
                      "cus_id": "비율(%)",
                      "grade": "고객 등급"
                     },)

fig.update_layout(title='고객별 등급 분포',font=dict(size=18,))
fig.show()


# 자산 규모별 투자 성향을 확인해 보았습니다.
# 
# 자산 규모가 클수록 위험을 감수하는 성향이 보이고, 작을수록 위험을 회피하는 성향이 보입니다.    
# 위에서 20대에서 30대로 넘어갈 때 투자성향의 분포가 바뀌는 게 자산 규모의 차이일지 확인해볼 필요성이 있습니다.   
# 
# 투자 성향을 다음과 같은 세 단계로 분류합니다.    
# 전문투자자형은 전체 데이터에서 2명이기 때문에 제외했습니다.    
# + 위험회피형(risk-averse): 안정형, 안정추구형
# + 위험중립형(risk-neutral) : 위험 중립형
# + 위험감수형(risk-loving) : 적극투자형, 공격투자형
# 
# 자산 규모도 다음과 같은 세 단계로 분류합니다.
# + 자산규모 하: ~ 자산 1000만원
# + 자산규모 중: 자산 1000만원 ~ 1억
# + 자산규모 상: 자산 1억 ~

# In[10]:


#2,30대 자산 규모 분류
cus_info['class'] = cus_info['grade'].replace({'01 탑클래스':'상','02 골드':'상','03 로얄':'상',
                                             '04 그린':'중', '05 블루':'중', '09 등급없음': '하'})

high_20 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['class'] == '상')]
middle_20 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['class'] == '중')]
low_20 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['class'] == '하')]
high_30 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['class'] == '상')]
middle_30 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['class'] == '중')]
low_30 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['class'] == '하')]


# In[11]:


risk = ['01 위험회피형','02 위험중립형','03 위험감수형']
color = ['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c']
label = label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 11)
h_20 = list(high_20.risk.value_counts().sort_index())
m_20 = list(middle_20.risk.value_counts().sort_index())
l_20 = list(low_20.risk.value_counts().sort_index())
h_30 = list(high_30.risk.value_counts().sort_index())
m_30 = list(middle_30.risk.value_counts().sort_index())
l_30 = list(low_30.risk.value_counts().sort_index())

tl = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(risk, l_20)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 20대")).set_series_opts(label))
tl.add(pie_0, "20대 자산 하위그룹")
pie_1 = (Pie().add("", [list(z) for z in zip(risk, m_20)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 20대")).set_series_opts(label))
tl.add(pie_1, "20대 자산 중위그룹")
pie_2 = (Pie().add("", [list(z) for z in zip(risk, h_20)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 20대")).set_series_opts(label))
tl.add(pie_2, "20대 자산 상위그룹")

tl2 = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(risk, l_30)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 30대")).set_series_opts(label))
tl2.add(pie_0, "30대 자산 하위그룹")
pie_1 = (Pie().add("", [list(z) for z in zip(risk, m_30)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 30대")).set_series_opts(label))
tl2.add(pie_1, "30대 자산 중위그룹")
pie_2 = (Pie().add("", [list(z) for z in zip(risk, h_30)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 30대")).set_series_opts(label))
tl2.add(pie_2, "30대 자산 상위그룹")
page = Page().add(tl,tl2)
page.render_notebook()


# 20대, 30대의 자산 규모 별 그룹에 따른 위험 회피정도를 시각화 해보았습니다.
# 
# 20대의 경우 자산이 적으면 위험을 회피하려고 하지만, 30대는 자산이 적더라도 위험을 감수하려고 합니다.   
# 20대와 30대 모두 자산 상위그룹의 경우 위험을 감수하고자 하며, 30대의 경우 그 비율이 더 높습니다.   
# 이를 통해 Y&Z세대 내에서 위험을 감수하는 정도는 나이가 많아질수록, 또 자산 규모가 커질수록 높아진다고 볼 수 있습니다.

# In[12]:


#계좌 개수 시각화

fig = px.strip(cus_info, x="age_cat", y="num_act", color="class",
                category_orders={"age_cat": ["10대 미만","20대","30대","40대","50대","60대","70대 이상"],
                                'class':['하','중','상']},
                labels={
                        "age_cat": "연령대",
                        "num_act": "개수(개)",
                       })
fig.update_layout(title='고객별 계좌 개수')
pyo.iplot(fig)


# 고객 별 계좌 개수를 시각화해보았습니다.
# 
# 나이가 많을수록 갖고 있는 계좌의 수가 많으며, 자산 규모가 커질수록 많음을 볼 수 있습니다.     
# 2,30대 내에서는 크게 차이가 두드러지진 않으나, 40대 이상에서는 그 차이가 극명합니다.

# In[13]:


act_cus.to_csv('act_cus.csv')
cus_info.to_csv('cus_info.csv')

