#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[20]:


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


# In[21]:


data_ind = pd.read_csv('data_ind.csv')


# ## 3.2 코로나 전후 그룹 특성 비교분석

# In[22]:


before = cus_info[cus_info['covid']=='before']
after = cus_info[cus_info['covid']=='after']


# In[23]:


#코로나 전후 등급 분포 시각화

grade = ['01 탑클래스','02 골드','03 로얄','04 그린','05 블루','09 등급없음']
before_grade = list(before.grade.value_counts().sort_index())
after_grade = list(after.grade.value_counts().sort_index())

tl = Timeline()
pie_0 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(grade, before_grade)],
            rosetype="radius",
            radius=["30%", "55%"],
        )
        .set_colors(['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c'])
        .set_global_opts(title_opts=opts.TitleOpts("고객 등급 분포: 코로나 이전"))
        .set_series_opts(label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)) 
)
tl.add(pie_0, "코로나 이전")
pie_1 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(grade, after_grade)],
            rosetype="radius",
            radius=["30%", "55%"],
        )
        .set_colors(['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c'])
        .set_global_opts(title_opts=opts.TitleOpts("고객 등급 분포: 코로나 이후"))
        .set_series_opts(label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)) 
)
tl.add(pie_1, "코로나 이후")
tl.render("timeline_pie.html")
tl.render_notebook()


# 코로나 이전에는 주로 자산이 어느 정도 있는 고객들(블루 이상)이 주식 거래에 참여하였다면,     
# 코로나 이후에는 자산이 상대적으로 적은 고객들(등급없음)의 참여가 매우 활발해졌습니다.
# 
# 코로나 이전에는 자산이 1억 미만이였던 사람들이 주식에 크게 관심이 없었지만, 주가폭락 이후 관심이 생긴 사람들이 많아졌다고 볼 수 있습니다.

# In[24]:


#코로나 전후 연령대 분포 시각화

grade = ['10대 미만','20대','30대','40대','50대','60대', '70대 이상']
before_age = list(before.age_cat.value_counts().sort_index())
after_age = list(after.age_cat.value_counts().sort_index())

tl = Timeline()
pie_0 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(grade, before_age)],
            rosetype="radius",
            radius=["30%", "55%"],
        )
        .set_colors(['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c'])
        .set_global_opts(title_opts=opts.TitleOpts("연령대 분포: 코로나 이전"))
        .set_series_opts(label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)) 
)
tl.add(pie_0, "코로나 이전")
pie_1 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(grade, after_age)],
            rosetype="radius",
            radius=["30%", "55%"],
        )
        .set_colors(['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c'])
        .set_global_opts(title_opts=opts.TitleOpts("연령대 분포: 코로나 이후"))
        .set_series_opts(label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)) 
)
tl.add(pie_1, "코로나 이후")
tl.render("timeline_pie.html")
tl.render_notebook()


# 연령대를 중심으로 살펴보았을 때도 코로나 이전과 이후에 큰 변화를 살펴볼 수 있습니다.     
# 코로나 이전에는 주로 40대, 50대 고객들이 지배적이었다면, 코로나 이후로는 20대, 30대고객들의 비중이 무척 증가하였습니다.      
# 특히 20대는 8.39%에서 18.89%로 폭발적인 증가를 보여주었습니다.

# ## 3.3 집단별 특성 비교

# 2번 모델링 과정에서 코로나 전후 진입 여부와 함께 성별이 투자성향에 끼치는 영향이 크다는 것을 확인했습니다.     
# 또한 많은 'Behavior Fincance' 연구에서 성별변수를 중요하게 여긴다는 것을 확인했습니다.  
# 그래서 저희는 집단을 다음과 같이 8가지로 나누어 여러 특징을 시각화해서 비교분석 해보았습니다.      
# 
# + 코로나 전 진입 20대 여자
# + 코로나 후 진입 20대 여자
# + 코로나 전 진입 20대 남자
# + 코로나 후 진입 20대 남자
# + 코로나 전 진입 30대 여자
# + 코로나 후 진입 30대 여자
# + 코로나 전 진입 30대 남자
# + 코로나 후 진입 30대 남자

# In[25]:


#집단 별로 데이터 자르기
before_ind = pd.merge(before, data_ind[['cus_id', 'iem_cd', 'iem_krl_nm', 'ind_cat']], on ='cus_id', how='left')
after_ind = pd.merge(after, data_ind[['cus_id', 'iem_cd', 'iem_krl_nm', 'ind_cat']], on ='cus_id', how='left')

male_20b = before_ind.loc[(before_ind['age_cat']=='20대') & (before_ind['sex'] == 1)]
female_20b = before_ind.loc[(before_ind['age_cat']=='20대') & (before_ind['sex'] == 2)]
male_30b = before_ind.loc[(before_ind['age_cat']=='30대') & (before_ind['sex'] == 1)]
female_30b = before_ind.loc[(before_ind['age_cat']=='30대') & (before_ind['sex'] == 2)]

male_20a = after_ind.loc[(after_ind['age_cat']=='20대') & (after_ind['sex'] == 1)]
female_20a = after_ind.loc[(after_ind['age_cat']=='20대') & (after_ind['sex'] == 2)]
male_30a = after_ind.loc[(after_ind['age_cat']=='30대') & (after_ind['sex'] == 1)]
female_30a = after_ind.loc[(after_ind['age_cat']=='30대') & (after_ind['sex'] == 2)]

male_20bb = before.loc[(before['age_cat']=='20대') & (before['sex'] == 1)]
female_20bb = before.loc[(before['age_cat']=='20대') & (before['sex'] == 2)]
male_30bb = before.loc[(before['age_cat']=='30대') & (before['sex'] == 1)]
female_30bb = before.loc[(before['age_cat']=='30대') & (before['sex'] == 2)]

male_20aa = after.loc[(after['age_cat']=='20대') & (after['sex'] == 1)]
female_20aa = after.loc[(after['age_cat']=='20대') & (after['sex'] == 2)]
male_30aa = after.loc[(after['age_cat']=='30대') & (after['sex'] == 1)]
female_30aa = after.loc[(after['age_cat']=='30대') & (after['sex'] == 2)]


# In[26]:


male_20bb['lala'] = '코로나 전 20대 남성'
male_20aa['lala'] = '코로나 후 20대 남성'
male_30bb['lala'] = '코로나 전 30대 남성'
male_30aa['lala'] = '코로나 후 30대 남성'
female_20bb['lala'] = '코로나 전 20대 여성'
female_20aa['lala'] = '코로나 후 20대 여성'
female_30bb['lala'] = '코로나 전 30대 여성'
female_30aa['lala'] = '코로나 후 30대 여성'


# In[27]:


from functools import reduce
import pandas as pd
dfs = [male_20bb,male_20aa,male_30bb,male_30aa,female_20bb,female_20aa,female_30bb,female_30aa]
df_merge = reduce(lambda left, right: pd.merge(left, right, on=['lala','cus_id','grade','class','risk','ivst'],how='outer'), dfs)


# In[28]:


#고객별 등급 분포 시각화
df_merge2 = df_merge.loc[(df_merge['ivst']!= '등급없음') & (df_merge['ivst']!='정보 제공 미동의')]
df = df_merge2.groupby(by=['lala', 'ivst']).count()
df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()

fig = px.bar(df, x='lala', y='cus_id', color='ivst',
             category_orders={'lala':['코로나 전 20대 남성','코로나 후 20대 남성','코로나 전 20대 여성','코로나 후 20대 여성','코로나 전 30대 남성','코로나 후 30대 남성','코로나 전 30대 여성','코로나 후 30대 여성'],
                             "ivst":['01 안정형','02 안정추구형','03 위험중립형','04 적극투자형','05 공격투자형','09 전문투자자형','등급없음','정보 제공 미동의']},
             labels={
                      "age_cat": "연령대",
                      "cus_id": "비율(%)",
                      "ivst": "고객 투자성향"
                     },)

fig.update_layout(title='군집별 투자성향 분포',font = dict(size=18))
fig.show()


# In[29]:


#코로나 전후 등급 분포 시각화

grade = ['01 탑클래스','02 골드','03 로얄','04 그린','05 블루','09 등급없음']
color = ['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c']
label = label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)
m_20b = list(male_20bb.grade.value_counts().sort_index())
f_20b = list(female_20bb.grade.value_counts().sort_index())
m_30b = list(male_30bb.grade.value_counts().sort_index())
f_30b = list(female_30bb.grade.value_counts().sort_index())
m_20a = list(male_20aa.grade.value_counts().sort_index())
f_20a = list(female_20aa.grade.value_counts().sort_index())
m_30a = list(male_30aa.grade.value_counts().sort_index())
f_30a = list(female_30aa.grade.value_counts().sort_index())

tl = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(grade, m_20b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 전 20대")).set_series_opts(label))
tl.add(pie_0, "코로나 전 20대 남성")
pie_1 = (Pie().add("", [list(z) for z in zip(['03 로얄','04 그린','05 블루','09 등급없음'], m_20a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 후 20대")).set_series_opts(label))
tl.add(pie_1, "코로나 후 20대 남성")
pie_2 = (Pie().add("", [list(z) for z in zip(grade, m_30b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 전 30대")).set_series_opts(label))
tl.add(pie_2, "코로나 전 30대 남성")
pie_3 = (Pie().add("", [list(z) for z in zip(grade, m_30a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 후 30대")).set_series_opts(label))
tl.add(pie_3, "코로나 후 30대 남성")


tl2 = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(['02 골드','03 로얄','04 그린','05 블루','09 등급없음'], f_20b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 전 20대")).set_series_opts(label))
tl2.add(pie_0, "코로나 전 20대 여성")
pie_1 = (Pie().add("", [list(z) for z in zip(['03 로얄','04 그린','05 블루','09 등급없음'], f_20a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 후 20대")).set_series_opts(label))
tl2.add(pie_1, "코로나 후 20대 여성")
pie_2 = (Pie().add("", [list(z) for z in zip(grade, f_30b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 전 30대")).set_series_opts(label))
tl2.add(pie_2, "코로나 전 30대 여성")
pie_3 = (Pie().add("", [list(z) for z in zip(['02 골드','03 로얄','04 그린','05 블루','09 등급없음'], f_30a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 등급: 코로나 후 30대")).set_series_opts(label))
tl2.add(pie_3, "코로나 후 30대 여성")

page = Page().add(tl,tl2)
page.render_notebook()


# 각 집단의 고객등급을 시각화 해보았습니다. 
# 
# 우선 코로나 이후 진입한 2,30대의 대다수가 자산이 1억 미만인 사람들임을 확인할 수 있습니다.    
# 그리고 전체적으로 여성보다 남성이, 20대가 30대보다 자산 규모가 작습니다.      
# 코로나 이전 20대의 등급분포와 코로나 이후 30대의 등급분포가 비슷한 것도 눈에 띕니다. 

# 고객 별 거래빈도를 시각화해보았습니다.
# 
# 집단간의 차이가 1 내외로, 크게 두드러지지 않습니다.

# In[30]:


#고객별 종목 산업 개수 시각화

df2 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_info.loc[(cus_info['age_cat']=='40대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.ind_cat, name='20대 남성',marker_color='#F3BE50')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.ind_cat, name='20대 여성',marker_color='#2C66B5')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.ind_cat, name='30대 남성',marker_color='#F3BE50')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.ind_cat, name='30대 여성',marker_color='#2C66B5')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 산업 개수'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)


# In[31]:


#고객별 종목 산업 개수 시각화

df2 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_info.loc[(cus_info['age_cat']=='40대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.freq, name='20대 남성',marker_color='#F3BE50')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.freq, name='20대 여성',marker_color='#2C66B5')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.freq, name='30대 남성',marker_color='#F3BE50')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.freq, name='30대 여성',marker_color='#2C66B5')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 거래빈도 (국내)'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)


# In[32]:


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


# 고객 별 상품 산업 개수를 시각화해보았습니다.
# 
# 우선 전체적으로 산업 개수는 코로나 이전 진입한 집단이 더 많습니다.    
# 또한 대체적으로 남성이 여성보다 보유하고 있는 산업의 개수가 더 많습니다.    
# 이를 통해 코로나 이전 진입한 집단이 조금 더 다양한 종목의 포트폴리오를 구성하고 있음을 볼 수 있습니다.

# In[33]:


#고객별 종목 산업 개수 시각화

df2 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df3 = cus_info.loc[(cus_info['age_cat']=='20대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df4 = cus_info.loc[(cus_info['age_cat']=='30대') & (cus_info['sex'] == 1)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()
df5 = cus_info.loc[(cus_info['age_cat']=='40대') & (cus_info['sex'] == 2)].groupby(by=['covid']).median().reindex(['before','after']).reset_index()

trace3 = go.Bar(x=['코로나 이전','코로나 이후'], y=df2.iem_num_kr, name='20대 남성',marker_color='#F3BE50')
trace4 = go.Bar(x=['코로나 이전','코로나 이후'], y=df3.iem_num_kr, name='20대 여성',marker_color='#2C66B5')
trace5 = go.Bar(x=['코로나 이전','코로나 이후'], y=df4.iem_num_kr, name='30대 남성',marker_color='#F3BE50')
trace6 = go.Bar(x=['코로나 이전','코로나 이후'], y=df5.iem_num_kr, name='30대 여성',marker_color='#2C66B5')

fig = go.Figure(data=[trace3, trace4, trace5, trace6], layout=go.Layout(title='고객별 종목 개수'))
fig.update_layout(font=dict(size=18,))
pyo.iplot(fig)


# 고객 별 상품 산업 개수를 시각화해보았습니다.
# 
# 종목의 개수는 코로나 이전 진입한 집단이 두배가랑 더 많습니다.    
# 또 코로나 이후 진입한 집단은 성별간 차이가 두드러지지 않으나,      
# 코로나 이전 진입한 집단의 경우 남성이 여성에 비해 더 많은 종목을 가지고 투자하고 있습니다.

# In[34]:


risk = ['01 위험회피형','02 위험중립형','03 위험감수형']
color = ['#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c']
label = label_opts=opts.LabelOpts(color=["black"], formatter="{b}:\n{d}% \n ", font_size = 15)
m_20b = list(male_20bb.risk.value_counts().sort_index())
f_20b = list(female_20bb.risk.value_counts().sort_index())
m_30b = list(male_30bb.risk.value_counts().sort_index())
f_30b = list(female_30bb.risk.value_counts().sort_index())
m_20a = list(male_20aa.risk.value_counts().sort_index())
f_20a = list(female_20aa.risk.value_counts().sort_index())
m_30a = list(male_30aa.risk.value_counts().sort_index())
f_30a = list(female_30aa.risk.value_counts().sort_index())

tl = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(risk, m_20b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 전 20대")).set_series_opts(label))
tl.add(pie_0, "코로나 전 20대 남성")
pie_1 = (Pie().add("", [list(z) for z in zip(risk, m_20a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 후 20대")).set_series_opts(label))
tl.add(pie_1, "코로나 후 20대 남성")
pie_2 = (Pie().add("", [list(z) for z in zip(risk, m_30b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 전 30대")).set_series_opts(label))
tl.add(pie_2, "코로나 전 30대 남성")
pie_3 = (Pie().add("", [list(z) for z in zip(risk, m_30a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 후 30대")).set_series_opts(label))
tl.add(pie_3, "코로나 후 30대 남성")


tl2 = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(risk, f_20b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 전 20대")).set_series_opts(label))
tl2.add(pie_0, "코로나 전 20대 여성")
pie_1 = (Pie().add("", [list(z) for z in zip(risk, f_20a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 후 20대")).set_series_opts(label))
tl2.add(pie_1, "코로나 후 20대 여성")
pie_2 = (Pie().add("", [list(z) for z in zip(risk, f_30b)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 전 30대")).set_series_opts(label))
tl2.add(pie_2, "코로나 전 30대 여성")
pie_3 = (Pie().add("", [list(z) for z in zip(risk, f_30a)], rosetype="radius",radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("고객 투자성향 분포: 코로나 후 30대")).set_series_opts(label))
tl2.add(pie_3, "코로나 후 30대 여성")

page = Page().add(tl,tl2)
page.render_notebook()


# 각 집단의 투자성향을 시각화 해보았습니다.
# 
# 우선 코로나 이후에 시장에 진입한 투자자들은 그 전에 진입한 투자자들에 비해 위험을 회피하려는 성향이 두드러집니다.    
# 또 전반적으로 20대보다 30대가, 여성보다 남성이 위험을 감수하려고 합니다.     
# 다시 말해 나이가 많을수록 위험을 감수하는 성향이 나타나며, 여성보다는 남성이 위험을 감수하고자 합니다.     
# 또 코로나 이후에는 위험 회피적인 투자자들이 많이 진입했습니다.

# In[35]:


m20b = list(male_20b.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
m20b_i = list(male_20b.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])
f20b = list(female_20b.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
f20b_i = list(female_20b.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])

m20a = list(male_20a.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
m20a_i = list(male_20a.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])
f20a = list(female_20a.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
f20a_i = list(female_20a.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])

m30b = list(male_30b.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
m30b_i = list(male_30b.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])
f30b = list(female_30b.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
f30b_i = list(female_30b.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])

m30a = list(male_30a.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
m30a_i = list(male_30a.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])
f30a = list(female_30a.iem_krl_nm.value_counts().sort_values(ascending=False)[0:10])
f30a_i = list(female_30a.iem_krl_nm.value_counts().sort_values(ascending=False).index[0:10])


# In[36]:


color = ['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c']
bar_20 = (
    Bar().add_xaxis(f20b_i).add_yaxis("코로나 전 20대 여자 Top10", f20b, color=[color[0]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 전 20대 여자", pos_top = '12%', pos_left='28%'),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_21 = (
    Bar().add_xaxis(f20a_i).add_yaxis("코로나 후 20대 여자 Top10", f20a, color=[color[1]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 후 20대 여자", pos_top = '12%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_30 = (
    Bar().add_xaxis(m20b_i).add_yaxis("코로나 전 20대 남자 Top10", m20b, color=[color[2]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 전 20대 남자",pos_top = '55%',  pos_left='28%'),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_31 = (
    Bar().add_xaxis(m20a_i).add_yaxis("코로나 후 20대 남자 Top10", m20a, color=[color[3]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 후 20대 남자",pos_top = '55%',  pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
grid1 = (
    Grid()
    .add(bar_20, grid_opts=opts.GridOpts(pos_bottom = '55%',  pos_right="55%"))
    .add(bar_21, grid_opts=opts.GridOpts(pos_bottom = '55%', pos_left="55%"))
    .add(bar_30, grid_opts=opts.GridOpts(pos_top = '55%', pos_right="55%"))
    .add(bar_31, grid_opts=opts.GridOpts(pos_top = '55%', pos_left="55%"))
)

bar_20 = (
    Bar().add_xaxis(f30b_i).add_yaxis("코로나 전 30대 여자 Top10", f30b, color=[color[4]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 전 30대 여자", pos_top = '12%', pos_left='28%'),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_21 = (
    Bar().add_xaxis(f30a_i).add_yaxis("코로나 후 30대 여자 Top10", f30a, color=[color[5]])
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 후 30대 여자", pos_top = '12%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_30 = (
    Bar().add_xaxis(m30b_i).add_yaxis("코로나 전 30대 남자 Top10", m30b)
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 전 30대 남자",pos_top = '55%',  pos_left='28%'),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))

bar_31 = (
    Bar().add_xaxis(m30a_i).add_yaxis("코로나 전 30대 남자 Top10", m30a)
    .set_global_opts(title_opts=opts.TitleOpts(title="코로나 후 30대 남자",pos_top = '55%',  pos_right='10%'),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))

grid2 = (
    Grid()
    .add(bar_20, grid_opts=opts.GridOpts(pos_bottom = '55%',  pos_right="55%"))
    .add(bar_21, grid_opts=opts.GridOpts(pos_bottom = '55%', pos_left="55%"))
    .add(bar_30, grid_opts=opts.GridOpts(pos_top = '55%', pos_right="55%"))
    .add(bar_31, grid_opts=opts.GridOpts(pos_top = '55%', pos_left="55%"))
)
page = Page().add(grid1,grid2)
page.render_notebook()


# 앞서 살펴보았듯, 남성은 보다 위험을 감수하는 공격적인 투자를 하고, 여성은 위험을 회피하는 안정적인 투자를하는 편인데,    
# 이러한 성향은 종목 선택에 있어서도  두드러지게 보여집니다.
# 
# **[20대 여성]**    
# 20대 여성은 비교적 안정적인 대기업에 대한 투자 비율이 높습니다.     
# 이는 가장 위험회피형의 비율이 높은 20대 여성의 투자 성향과 관련이 있어 보입니다.   
# 코로나 이전과 이후의 상위 10개 종목에 시가 총액 10위 이내의 기업들이 많이 포함 되어있습니다.    
# 눈에 띄는 종목의 변화는 NAVER보통주와 삼성전자우선주가 코로나 이후 새롭게 10위권 내에 등장했다는 점입니다.     
# 코로나 이후 IT부문에 대한 관심이 증가했다고 생각됩니다.   
# 
# **[20대 남성]**    
# 20대 남성은 어느정도 위험감수형이 있는 만큼, 코로나 이전과 이후의 상위 10개 종목에서 삼성 KODEX 200선물 인버스 2X증권상장지수투자신탁(주식-파생형)을 찾아볼 수 있습니다.     
# 삼성 KODEX 200선물 인버스 2X는 시장 하락분의 2배 수익을 노리는 상품으로 베팅을 해야하는 종목입니다.     
# 이는 20대 남성의 공격적인 투자 성향을 보여주며, 코로나 이후 이 종목을 보유한 비율이 이전보다 높아졌습니다.    
# 
# **[30대 여성]**     
# 30대 여성은 20대 여성에 비해 비교적 규모가 작은 기업에 대해서도 높은 투자 비율을 보여주고 있습니다.         
# 현재의 규모와 안정성 못지 않게 기업의 성장가능성을 높게 본다고 추측해볼 수 있을 것 같습니다.    
# 특히 코로나 이후, 의료 부문의 투자 비율이 높아진 점이 눈에 띕니다.      
# 관련 종목이 무려 4개나 상위 10개 종목에 포함되어 있습니다.    
# 이는 코로나 상황의 영향으로 보입니다.
# 
# **[30대 남성]**      
# 가장 위험을 감수하는 성향인 30대 남성의 상위 10개 종목에서도 삼성 KODEX 200선물 인버스 2X증권상장지수투자신탁(주식-파생형)를
# 찾아볼 수 있었습니다.     
# 이 역시 30대 남성의 공격적인 투자 성향을 보여준다고 사료됩니다.     
# 또한, 인버스 상품에 대한 투자 비율이 늘었다는 점에서, 코로나 이후로 더욱더 공격적인 투자 성향을 보이고 있다고 판단할 수 있습니다.    
# 그리고 코로나 이후 의료부문 투자 비율이 무척 높아졌습니다.

# In[37]:


m20b = list(male_20b.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
m20b_i = list(male_20b.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])
f20b = list(female_20b.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
f20b_i = list(female_20b.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])

m20a = list(male_20a.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
m20a_i = list(male_20a.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])
f20a = list(female_20a.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
f20a_i = list(female_20a.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])

m30b = list(male_30b.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
m30b_i = list(male_30b.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])
f30b = list(female_30b.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
f30b_i = list(female_30b.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])

m30a = list(male_30a.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
m30a_i = list(male_30a.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])
f30a = list(female_30a.ind_cat_y.value_counts().sort_values(ascending=False)[0:10])
f30a_i = list(female_30a.ind_cat_y.value_counts().sort_values(ascending=False).index[0:10])


# In[38]:


color = ['#fc6472', '#f4b2a6', '#eccdb3', '#bcefd0', '#a1e8e4', '#23c8b2', '#7f5a7c']
label = label_opts=opts.LabelOpts(color=["black"], formatter="{d}% \n {b}", font_size = 11)

tl = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(m20b_i, m20b)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("20대 남성 투자 산업: 코로나 이전"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl.add(pie_0, "코로나 이전")
pie_1 = (Pie().add("", [list(z) for z in zip(m20a_i, m20a)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("20대 남성 투자 산업: 코로나 이후"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl.add(pie_1, "코로나 이후")
pie_2 = (Pie().add("", [list(z) for z in zip(m30b_i, m30b)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("30대 남성 투자 산업: 코로나 이전"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl.add(pie_2, "코로나 이전")
pie_3 = (Pie().add("", [list(z) for z in zip(m30a_i, m30a)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("30대 남성 투자 산업: 코로나 이후"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl.add(pie_3, "코로나 이후")


tl2 = Timeline()
pie_0 = (Pie().add("", [list(z) for z in zip(f20b_i, f20b)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("20대 여성 투자 산업: 코로나 이전"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl2.add(pie_0, "코로나 이전")
pie_1 = (Pie().add("", [list(z) for z in zip(f20a_i, f20a)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("20대 여성 투자 산업: 코로나 이후"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl2.add(pie_1, "코로나 이후")
pie_2 = (Pie().add("", [list(z) for z in zip(f30b_i, f30b)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("30대 여성 투자 산업: 코로나 이전"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl2.add(pie_2, "코로나 이전")
pie_3 = (Pie().add("", [list(z) for z in zip(f30a_i, f30a)], radius=["30%", "55%"])
        .set_colors(color).set_global_opts(title_opts=opts.TitleOpts("30대 여성 투자 산업: 코로나 이후"),
                                          legend_opts=opts.LegendOpts(is_show = True, orient='vertical', pos_left='left', pos_top ='middle'))
         .set_series_opts(label))
tl2.add(pie_3, "코로나 이후")

page = Page().add(tl,tl2)
page.render_notebook()


# 각 집단별 산업 분포를 시각화 해보았습니다.
# 

# In[39]:


cus_info2 = cus_info[['cus_id','age_cat','sex','covid','grade','freq','iem_num_kr']]
cus_info2.to_csv('cus_info2.csv')

