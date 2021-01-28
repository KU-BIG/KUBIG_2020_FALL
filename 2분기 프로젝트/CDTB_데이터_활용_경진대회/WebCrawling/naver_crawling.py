from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver import ActionChains

c_list = pd.read_csv('./cafe_list.csv', encoding='euc-kr')
for i in range(len(c_list["nlink"])):
    if i < 48:
        continue
    # 크롬 드라이버 로딩
    driver = webdriver.Chrome('./chromedriver.exe')
    driver.implicitly_wait(3)
    # 크롤링할 주소 입력
    url = c_list["nlink"][i]
    driver.get(url)

    time.sleep(5)

    # # id가 something 인 element 를 찾음
    # some_tag = driver.find_element_by_class_name('_3iTUo')
    #
    # # somthing element 까지 스크롤
    # action = ActionChains(driver)
    # action.move_to_element(some_tag).perform()
    iframes = driver.find_elements_by_tag_name('iframe')
    driver.switch_to.frame("entryIframe")

    # 리뷰 저장할 메모장 파일
    file = open('result/%s.txt'%c_list["name"][i], 'w', encoding="utf-8")  # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환
    while True:
        try:
            button = driver.find_element_by_class_name("_3iTUo")
            button.click()
            time.sleep(1)
            # print("%d번째 더보기 클릭" % i)
            # i += 1
        except:
            print("더보기 끝")
            break
    reviews = driver.find_elements_by_class_name("WoYOw")
    print("총 ", len(reviews), "개의 리뷰")
    result = []
    for review in reviews:
        file.write(review.text + '\n')
    file.close()
    driver.close()

