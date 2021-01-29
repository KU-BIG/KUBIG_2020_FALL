#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[16]:


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


# ## 1.5 주식 상품 데이터

# In[4]:


#상장법인목록 불러오기
ind_code = pd.read_csv('상장법인목록.csv', engine = 'python', dtype = 'str').reset_index()
iem = pd.read_csv('2_iem_info.csv')

#column명 바꾸기
ind_code.rename(columns = {'level_2':'industry', 'level_1':'stock_code'}, inplace=True)
ind_code['stock_code']=ind_code['stock_code'].astype(str)
ind_code['stock_code']=ind_code['stock_code'].str.zfill(6)

#종목코드에서 숫자만 추출 (상장법인목록과 비교 위해)
iem["stock_code"] = iem["iem_cd"].str[1:7]

#종목코드를 기준으로 item과 ind_code merge
item_n = pd.DataFrame(pd.merge(iem, ind_code, on='stock_code', how='left'))

#불필요한 열 제거
item_n = item_n[['iem_cd', 'iem_eng_nm', 'iem_krl_nm', 'industry']]


# 상장법인목록에서 구별된 업종의 개수가 너무 많아, 이를 10개의 산업으로 분류했습니다.  
# 분류 기준은 기업 재무정보 대표 업체인 에프앤가이드에서 만든 Fn universe라는 분류체계를 따랐습니다.  
# * 산업 분류: 에너지, 소재, 산업재, 경기소비재, 필수소비재, 의료, 금융, IT, 통신서비스, 유틸리티

# In[5]:


에너지 = ['연료용 가스 제조 및 배관공급업','기타 섬유제품 제조업', '석유 정제품 제조업', '해체, 선별 및 원료 재생업', '석탄 광업', '전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업']

소재 = ['자연과학 및 공학 연구개발업', '유리 및 유리제품 제조업', '기초 화학물질 제조업', '골판지, 종이 상자 및 종이용기 제조업',  '기타 화학제품 제조업',  '펄프, 종이 및 판지 제조업', '기타 종이 및 판지 제품 제조업',
 '플라스틱제품 제조업', '합성고무 및 플라스틱 물질 제조업', '특수 목적용 기계 제조업', '기타 비금속광물 광업', '1차 비철금속 제조업',  '기타 금속 가공제품 제조업', '금속 주조업',  '기타 비금속 광물제품 제조업', '화학섬유 제조업','나무제품 제조업', '고무제품 제조업',  '무기 및 총포탄 제조업', '내화, 비내화 요업제품 제조업',
 '제재 및 목재 가공업', '가구 제조업', '시멘트, 석회, 플라스터 및 그 제품 제조업']

산업재 = ['절연선 및 케이블 제조업', '선박 및 보트 건조업', '그외 기타 제품 제조업', '1차 철강 제조업', '해상 운송업', '건물 건설업',  '육상 여객 운송업', '기타 운송관련 서비스업',
 '기반조성 및 시설물 축조관련 전문공사업',  '철도장비 제조업', '실내건축 및 건축마무리 공사업', '건축자재, 철물 및 난방장치 도매업', '건물설비 설치 공사업','도로 화물 운송업', '자동차 판매업', '산업용 기계 및 장비 임대업', 
 '구조용 금속제품, 탱크 및 증기발생기 제조업','운송장비 임대업', '기타 전기장비 제조업', '도축, 육류 가공 및 저장 처리업',  '항공 여객 운송업', '그외 기타 운송장비 제조업', '기타 과학기술 서비스업', '항공기,우주선 및 부품 제조업', 
 '그외 기타 전문, 과학 및 기술 서비스업', '건축기술, 엔지니어링 및 관련 기술 서비스업', '폐기물 처리업',  '측정, 시험, 항해, 제어 및 기타 정밀기기 제조업; 광학기기 제외',
 '증기, 냉·온수 및 공기조절 공급업',  '일반 목적용 기계 제조업', '전기업', '토목 건설업']

경기소비재 = ['자동차 재제조 부품 제조업', '상품 중개업', '섬유, 의복, 신발 및 가죽제품 소매업','상품 종합 도매업', '봉제의복 제조업', '여행사 및 기타 여행보조 서비스업',  '기타 상품 전문 소매업',  '교육지원 서비스업', 
 '일반 교습 학원',   '텔레비전 방송업', '오디오물 출판 및 원판 녹음업', '창작 및 예술관련 서비스업', '경비, 경호 및 탐정업',   '초등 교육기관', '기타 교육기관',  '회사 본부 및 경영 컨설팅 서비스업',
 '광고업', '기타 전문 서비스업',  '일반 및 생활 숙박시설 운영업', '기타 사업지원 서비스업', '전문디자인업',  '유원지 및 기타 오락관련 서비스업', '종합 소매업'
 '음식점업', '그외 기타 개인 서비스업', '직물직조 및 직물제품 제조업', '무점포 소매업', '편조원단 제조업', '섬유제품 염색, 정리 및 마무리 가공업', '기타 전문 도매업', 
 '스포츠 서비스업', '산업용 농·축산물 및 동·식물 도매업',  '사업시설 유지·관리 서비스업', '자동차 신품 부품 제조업', '자동차용 엔진 및 자동차 제조업',
 '가전제품 및 정보통신장비 소매업',  '영화, 비디오물, 방송프로그램 제작 및 배급업', '자동차 부품 및 내장품 판매업',  '연료 소매업','방적 및 가공사 제조업', 
 '기계장비 및 관련 물품 도매업',  '가죽, 가방 및 유사제품 제조업', '신발 및 신발 부분품 제조업', '악기 제조업',  '귀금속 및 장신용품 제조업', '인쇄 및 인쇄관련 산업', '서적, 잡지 및 기타 인쇄물 출판업',
 '전구 및 조명장치 제조업','기타 섬유제품 제조업', '기타 생활용품 소매업', '음식점업', '종합 소매업', '가정용 기기 제조업']

필수소비재 = ['음·식료품 및 담배 도매업', '담배 제조업', '생활용품 도매업', '개인 및 가정용품 임대업', '기타 생활용품 소매업'
 '가정용 기기 제조업',  '비알코올음료 및 얼음 제조업', '알코올음료 제조업', '동물용 사료 및 조제식품 제조업', 
 '곡물가공품, 전분 및 전분제품 제조업', '과실, 채소 가공 및 저장 처리업', '작물 재배업', '기타 식품 제조업', '수산물 가공 및 저장 처리업',
 '낙농제품 및 식용빙과류 제조업']

의료 = ['의약품 제조업', '기초 의약물질 및 생물학적 제제 제조업', '의료용품 및 기타 의약 관련제품 제조업',
 '의료용 기기 제조업',  '비료, 농약 및 살균, 살충제 제조업', '환경 정화 및 복원업', '어로 어업']

금융 = ['기타 금융업', '보험업',  '은행 및 저축기관',  '부동산 임대 및 공급업',
'재 보험업',  '신탁업 및 집합투자업', '금융 지원 서비스업']

IT = ['통신 및 방송 장비 제조업', '전자부품 제조업', '자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업', '소프트웨어 개발 및 공급업', '반도체 제조업', '컴퓨터 프로그래밍, 시스템 통합 및 관리업',
'컴퓨터 및 주변장치 제조업',  '기타 정보 서비스업', '기록매체 복제업', '영상 및 음향기기 제조업',  '사진장비 및 광학기기 제조업', '일차전지 및 축전지 제조업']

통신서비스 = ['전기 통신업', '전기 및 통신 공사업']


# In[6]:


def ind(x) :
    if x in 에너지:
        return "에너지"
    elif x in 소재:
        return "소재"
    elif x in 산업재:
        return "산업재"
    elif x in 경기소비재:
        return "경기소비재"
    elif x in 필수소비재:
        return "필수소비재"
    elif x in 의료:
        return "의료"
    elif x in 금융:
        return "금융"
    elif x in IT:
        return "IT"
    elif x in 통신서비스:
        return "통신서비스"
    else:
        return "기타"
    
item_n["ind_cat"] = item_n["industry"].apply(lambda x : ind(x))


# In[7]:


#회사이름에 투자나 증권이 들어가면 금융으로 분류
for i in range(0, len(item_n)):
    if item_n.ind_cat[i] == '기타':
        if '투자' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '금융'
            item_n.industry[i] = '신탁업 및 집합투자업'
        elif '증권' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '금융'
            item_n.industry[i] = '신탁업 및 집합투자업'
        else: item_n.ind_cat[i] = '기타'
    else:
        item_n.ind_cat[i] = item_n.ind_cat[i]


# In[8]:


#우선주라 분류가 되지 않은 종목 분류
for i in range(0, len(item_n)):
    if item_n.ind_cat[i] == '기타':
        if '자동차' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '경기소비재'
        elif '전자' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = 'IT'
        elif '성신양회' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '소재'
        elif '대원전선' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '서울식품' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '크라운' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '중공업' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '화학' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '소재'
        elif '전기' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = 'IT'
        elif '삼성SDI' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = 'IT'
        elif '보험' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '금융'
        elif '은행' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '금융'
        elif '푸드' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '약품' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '의료'
        elif '항공' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'   
        elif '한진' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '대교' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '경기소비재'
        elif '바이오' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '의료'
        elif '남양' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '미래에셋' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '금융'
        elif '건설' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '두산' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '타이어' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '소재'   
        elif '페이퍼' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '소재'
        elif '대림' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '산업재'
        elif '에스마크' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = 'IT'
        elif 'Oil' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '에너지'
        elif '소재' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '소재'   
        elif '유니슨' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '에너지'
        elif '사조해표' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '생활건강' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '진로' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '칠성' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '필수소비재'
        elif '제약' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '의료'
        elif '호텔' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = '경기소비재'
        elif '정보' in item_n.iem_krl_nm[i]:
            item_n.ind_cat[i] = 'IT'
        else: item_n.ind_cat[i] = '기타'
    else:
        item_n.ind_cat[i] = item_n.ind_cat[i]


# In[17]:


df = pd.merge(trdkr, item_n, on='iem_cd')

#필요한 column만 추출
df = df[['act_id','iem_cd','net_purchase_kr','iem_krl_nm','ind_cat']]
#고객별 순매수금액 비중 구하기
net = pd.merge(df, act_cus, on=['act_id']).groupby(by = ['cus_id','iem_cd']).sum()
net = net.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()
#종목 분류별 순매수금액 비중 구하기
cat_net = pd.merge(net,item_n, on='iem_cd')
cat_net = cat_net.groupby(by = ['cus_id','ind_cat']).sum()
cat_net = cat_net.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()
cat = cat_net['ind_cat'].unique().tolist()
df2 = pd.DataFrame(cus_info['cus_id'])
for i in cat:
    df1 = cat_net[cat_net['ind_cat']==i][['cus_id','net_purchase_kr']]
    df1.rename(columns ={'net_purchase_kr':i}, inplace = True)
    df2 = pd.merge(df2, df1, on='cus_id', how='outer')
df2 = df2.fillna(0)

#비중 cus_info 데이터에 합쳐주기
cus_info = pd.merge(cus_info, df2, on='cus_id')


# In[18]:


df = pd.merge(trdkr, item_n, on='iem_cd')
df = df[['act_id','iem_cd','net_purchase_kr','iem_krl_nm','ind_cat']]
df = pd.merge(act_cus, df, on = 'act_id')
data_ind = pd.merge(cus_info, df, on = 'cus_id')

drop_20 = data_ind[data_ind['age_cat'] == '20대']
drop_30 = data_ind[data_ind['age_cat'] == '30대']
drop_40 = data_ind[data_ind['age_cat'] == '40대']
drop_50 = data_ind[data_ind['age_cat'] == '50대']
drop_60 = data_ind[data_ind['age_cat'] == '60대']
drop_70 = data_ind[data_ind['age_cat'] == '70대 이상']


# In[20]:


color = col


# In[21]:


#연령대별 top10 인기 종목 시각화

count_20 = list(drop_20.iem_krl_nm.value_counts())[:10]
stock_20 = list(drop_20.iem_krl_nm.value_counts().index)[0:10]
count_30 = list(drop_30.iem_krl_nm.value_counts())[:10]
stock_30 = list(drop_30.iem_krl_nm.value_counts().index)[0:10]
count_40 = list(drop_40.iem_krl_nm.value_counts())[:10]
stock_40 = list(drop_40.iem_krl_nm.value_counts().index)[0:10]
count_50 = list(drop_50.iem_krl_nm.value_counts())[:10]
stock_50 = list(drop_50.iem_krl_nm.value_counts().index)[0:10]
count_60 = list(drop_60.iem_krl_nm.value_counts())[:10]
stock_60 = list(drop_60.iem_krl_nm.value_counts().index)[0:10]
count_70 = list(drop_70.iem_krl_nm.value_counts())[:10]
stock_70 = list(drop_70.iem_krl_nm.value_counts().index)[0:10]

bar_20 = (
    Bar().add_xaxis(stock_20).add_yaxis("20대 Top10", count_20, color=color[0])
    .set_global_opts(title_opts=opts.TitleOpts(title="20대", pos_top = '5%', pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_30 = (
    Bar().add_xaxis(stock_30).add_yaxis("30 대 Top10", count_30, color=color[2])
    .set_global_opts(title_opts=opts.TitleOpts(title="30대", pos_top = '5%',pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_40 = (
    Bar().add_xaxis(stock_40).add_yaxis("40대 Top10", count_40, color=color[2])
    .set_global_opts(title_opts=opts.TitleOpts(title="40대", pos_top = '39%', pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_50 = (
    Bar().add_xaxis(stock_50).add_yaxis("50 대 Top10", count_50, color=color[3])
    .set_global_opts(title_opts=opts.TitleOpts(title="50대", pos_top = '39%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_60 = (
    Bar().add_xaxis(stock_60).add_yaxis("60대 Top10", count_60, color=color[4])
    .set_global_opts(title_opts=opts.TitleOpts(title="60대",pos_top = '68%',  pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_70 = (
    Bar().add_xaxis(stock_70).add_yaxis("70 대 Top10", count_70, color=color[5])
    .set_global_opts(title_opts=opts.TitleOpts(title="70대", pos_top = '68%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                     legend_opts=opts.LegendOpts(is_show = False))) 
grid = (
    Grid()
    .add(bar_20, grid_opts=opts.GridOpts(pos_top = '10%', pos_bottom = '70%', pos_right="55%"))
    .add(bar_30, grid_opts=opts.GridOpts(pos_top = '10%', pos_bottom = '70%', pos_left="55%"))
    .add(bar_40, grid_opts=opts.GridOpts(pos_top = '40%', pos_bottom = '40%', pos_right="55%"))
    .add(bar_50, grid_opts=opts.GridOpts(pos_top = '40%', pos_bottom = '40%', pos_left="55%"))
    .add(bar_60, grid_opts=opts.GridOpts(pos_top = '67%', pos_right="55%"))
    .add(bar_70, grid_opts=opts.GridOpts(pos_top = '67%', pos_left="55%"))
)
grid.render_notebook()


# 연령대별 TOP 10 인기 종목을 시각화 해보았습니다.
# 
# 우선 대체적으로 제약관련 종목이 인기가 많음을 확인할 수 있습니다. 
# 이는 저희 주제와 마찬가지로 코로나가 주식시장에  전 연령대에 있어 영향을 주고 있음을 의미하고 있습니다. 
# 
# 주목해서 볼만한 점은, 20대의 인기 종목이 살짝 다르다는 점입니다.   
# 우선 상위 10개 종목 중 유일하게 우선주 투자가 이루어진 항목이 있습니다.        
# 아무래도 대학생, 사회초년생의 비율이 높다보니 의사결정권 대신 보통주보다 높은 수익률을 보장하는 우선주에 높은 관심을 보이는 것 같습니다.
# 또 시가총액이 높은 대형주의 비율이 타 연령층에 비해 높습니다.    
# 이는 주식을 시작한지 얼마 되지 않아 위험회피적인 성향을 띈다고 볼 수 있습니다.    
# 실제로 20대에 위험 회피적인 성향을 띄는 투자자가 절반 가까이 되었습니다.
# 
# **[시가총액 Top 10 기업]**    
# 삼성전자  
# SK하이닉스  
# 삼성전자우  
# LG화학  
# 삼성바이오로직스  
# 셀트리온  
# NAVER  
# 삼성SDI  
# 현대차  
# 카카오  
# 
# 
# 

# In[22]:


#보유 종목 개수로 순위 계산
high = drop_20[drop_20['class'] == '상']
mid = drop_20[drop_20['class'] == '중']
low = drop_20[drop_20['class'] == '하']

count_high = list(high.iem_krl_nm.value_counts())[:10]
stock_high = list(high.iem_krl_nm.value_counts().index)[0:10]
count_mid = list(mid.iem_krl_nm.value_counts())[:10]
stock_mid = list(mid.iem_krl_nm.value_counts().index)[0:10]
count_low = list(low.iem_krl_nm.value_counts())[:10]
stock_low = list(low.iem_krl_nm.value_counts().index)[0:10]

bar_21 = (
    Bar().add_xaxis(stock_high).add_yaxis("20대 상위그룹 Top10", count_high, color=[color[0]])
    .set_global_opts(title_opts=opts.TitleOpts(title="20대 상", pos_top = '5%', pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_22 = (
    Bar().add_xaxis(stock_mid).add_yaxis("20대 중위그룹 Top10", count_mid, color=[color[1]])
    .set_global_opts(title_opts=opts.TitleOpts(title="20대 중", pos_top = '39%', pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_23 = (
    Bar().add_xaxis(stock_low).add_yaxis("20대 하위그룹 Top10", count_low, color=[color[2]])
    .set_global_opts(title_opts=opts.TitleOpts(title="20대 하",pos_top = '68%',  pos_left="38%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))


# In[23]:


high = drop_30[drop_30['class'] == '상']
mid = drop_30[drop_30['class'] == '중']
low = drop_30[drop_30['class'] == '하']

count_high = list(high.iem_krl_nm.value_counts())[:10]
stock_high = list(high.iem_krl_nm.value_counts().index)[0:10]
count_mid = list(mid.iem_krl_nm.value_counts())[:10]
stock_mid = list(mid.iem_krl_nm.value_counts().index)[0:10]
count_low = list(low.iem_krl_nm.value_counts())[:10]
stock_low = list(low.iem_krl_nm.value_counts().index)[0:10]

bar_31 = (
    Bar().add_xaxis(stock_high).add_yaxis("30대 상위그룹 Top10", count_high, color=[color[3]])
    .set_global_opts(title_opts=opts.TitleOpts(title="30대 상", pos_top = '5%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_32 = (
    Bar().add_xaxis(stock_mid).add_yaxis("30대 중위그룹 Top10", count_mid, color=[color[4]])
    .set_global_opts(title_opts=opts.TitleOpts(title="30대 중", pos_top = '39%', pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
bar_33 = (
    Bar().add_xaxis(stock_low).add_yaxis("30대 하위그룹 Top10", count_low, color=[color[5]])
    .set_global_opts(title_opts=opts.TitleOpts(title="30대 하",pos_top = '68%',  pos_right="10%"),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show = False)),
                     legend_opts=opts.LegendOpts(is_show = False)))
grid = (
    Grid()
    .add(bar_21, grid_opts=opts.GridOpts(pos_top = '10%', pos_bottom = '70%', pos_right="55%"))
    .add(bar_31, grid_opts=opts.GridOpts(pos_top = '10%', pos_bottom = '70%', pos_left="55%"))
    .add(bar_22, grid_opts=opts.GridOpts(pos_top = '40%', pos_bottom = '40%', pos_right="55%"))
    .add(bar_32, grid_opts=opts.GridOpts(pos_top = '40%', pos_bottom = '40%', pos_left="55%"))
    .add(bar_23, grid_opts=opts.GridOpts(pos_top = '67%', pos_right="55%"))
    .add(bar_33, grid_opts=opts.GridOpts(pos_top = '67%', pos_left="55%"))
)
grid.render_notebook()


# **[20대]** 
# 
# 자산 규모 별로 구분한 20대 고객의 상위 10개 종목은 상위 그룹, 중위 그룹, 하위 그룹 모두 전반적으로 비슷합니다. 공통적으로 삼성전자보통주가 가장 높은 비중을 차지하고 있으며, 시가총액 10위 이내의 기업들이 많이 포함되어 있습니다.
#  
#  그러나 눈 여겨봐야할 점은 각 종목들의 가격입니다. 상위 그룹에만 있는 삼성바이오로직스보통주, 삼성SDI보통주, LG화학보통주의 가격은 각각 82,6000원, 62,8000원, 82,4000원으로 상당히 높습니다. 중위 그룹에서는 30만원 후반대의 카카오보통주, 셀트리온보통주가 가장 가격이 높으며, 하위 그룹에서 30만원 이상의 종목은 카카오보통주만 존재합니다.
#  
#  자산 규모가 큰 20대 고객들은 높은 가격의 종목들도 많이 보유하고 있지만, 자산 규모가 천 만원 이내의 고객들은 대체로 10만원 전 후의 비교적 저렴한 종목들을 주로 보유하고 있습니다.
#  
# **[30대]**    
# 
#  자산 규모 별로 구분한 30대 고객의 상위 10개 종목도 20대와 비슷하게 삼성전자보통주가 가장 높은 비중을 차지하고 있습니다. 그러나 20대와 다르게 상위 그룹, 중위 그룹, 하위 그룹의 종목들이 가격적인 측면에서 유의미한 차이가 있지는 않습니다. 하지만 전체적인 종목의 구성에서 여러 비교를 해볼 수가 있습니다.
#  
#  먼저, 시가총액이 10위 이내의 기업들이 20대 상위 종목에는 많이 포함되어 있는 반면, 30대 상위 종목에는 비교적 적게 포함되어 있습니다. 이는 20대는 안전추구형이 많고, 30대는 위험을 감수하는 성향이 많은 것과 관련하여 생각해볼 수 있습니다.
#  
#  또한, 20대와 다르게 제약, 바이오 관련 종목들이 많이 포함되어 있습니다.     
#  이는 코로나19 상황과 관련하여 생각해볼 수 있을 것 같아 밑에서 더 논의하도록 하겠습니다.
#  
# - 상위 그룹: 셀트리온보통주, 씨젠, 신라젠, 셀트리온헬스케어, 파미셀보통주       
# - 중위 그룹: 파미셀보통주, 신풍제약보통주, 씨젠, 이원다이애그노믹스            
# - 하위 그룹: 씨젠, 신풍제약보통주, 파미셀              

# In[24]:


df = data_ind[['cus_id', 'ind_cat']]
df = df.drop_duplicates()
df = df.groupby('cus_id').count().reset_index()
cus_info = pd.merge(cus_info, df, on ='cus_id', how = 'outer').fillna(0)

#고객별 종목산업개수 시각화
df2 = cus_info.loc[cus_info['age_cat'] == '20대'].groupby(by=['grade']).median().reset_index()
df3 = cus_info.loc[cus_info['age_cat'] == '30대'].groupby(by=['grade']).median().reset_index()
df4 = cus_info.loc[cus_info['age_cat'] == '40대'].groupby(by=['grade']).median().reset_index()
df5 = cus_info.loc[cus_info['age_cat'] == '50대'].groupby(by=['grade']).median().reset_index()
df6 = cus_info.loc[cus_info['age_cat'] == '60대'].groupby(by=['grade']).median().reset_index()
df7 = cus_info.loc[cus_info['age_cat'] == '70대 이상'].groupby(by=['grade']).median().reset_index()

trace3 = go.Bar(x=df2.grade, y=df2.ind_cat, name='20대',marker_color=col[0])
trace4 = go.Bar(x=df3.grade, y=df3.ind_cat, name='30대',marker_color=col[1])
trace5 = go.Bar(x=df4.grade, y=df4.ind_cat, name='40대',marker_color=col[2])
trace6 = go.Bar(x=df5.grade, y=df5.ind_cat, name='50대',marker_color=col[3])
trace7 = go.Bar(x=df6.grade, y=df6.ind_cat, name='60대',marker_color=col[4])
trace8 = go.Bar(x=df7.grade, y=df7.ind_cat, name='70대',marker_color=col[5])

fig = go.Figure(data=[trace3, trace4, trace5, trace6, trace7,trace8])
fig.update_layout(title='고객별 산업 개수',font=dict(size=18,))
pyo.iplot(fig)


# 고객별 투자 포트폴리오의 다양성을 알아보기 위해 산업의 개수를 시각화해보았습니다.
# 
# 확실히 자산 규모가 높은 탑클래스의 경우 종목의 개수가 많은만큼 투자하는 산업의 가지수도 다양합니다.     
# 20대 탑클래스의 경우 평균적으로 모든 산업의 상품을 가지고 투자하고 있었습니다.         
# 포트폴리오가 다양할수록 자산규모가 높아짐을 확인할 수 있습니다.
