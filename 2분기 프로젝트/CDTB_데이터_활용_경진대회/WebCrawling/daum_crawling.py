from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver import ActionChains

c_list = pd.read_csv('./cafe_list.csv', encoding='euc-kr')
for i in range(len(c_list["dlink"])):
    if i < 55 :
        continue
    # 크롬 드라이버 로딩
    driver = webdriver.Chrome('./chromedriver.exe')
    driver.implicitly_wait(3)
    # 크롤링할 주소 입력
    url = c_list["dlink"][i]
    driver.get(url)
    time.sleep(3)
    # 리뷰 저장할 메모장 파일
    file = open('result/%s.txt'%c_list["name"][i], 'w', encoding="utf-8")  # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환
    pages = driver.find_elements_by_class_name("link_page")
    result = []
    for page in range(len(pages)):
        pages = driver.find_elements_by_class_name("link_page")
        try:
            pages[page].click()
        except:
            print("Error")
        time.sleep(1)
        reviews = driver.find_elements_by_class_name("txt_comment")
        for review in reviews:
            # print(review.text)
            file.write(review.text + '\n')
    file.close()
    driver.close()

