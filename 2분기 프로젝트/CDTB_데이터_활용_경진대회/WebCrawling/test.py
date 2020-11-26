from selenium import webdriver
from urllib.request import urlopen # 웹서버에 접근 모듈
from bs4 import BeautifulSoup # 웹페이지 내용구조 분석 모듈
url = 'https://www.naver.com/'
html=urlopen(url)
html_source = BeautifulSoup(html, 'html.parser')
print(html_source)
driver = webdriver.Chrome('./chromedriver.exe')
driver.get(url)
driver.implicitly_wait(10)
driver.find_element_by_xpath('//*[@id="NM_NEWSSTAND_HEADER"]/div[2]/a[3]').click()