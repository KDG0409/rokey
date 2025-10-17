# requests 모듈과 pandas 모듈, bs4 모듈의 BeautifulSoup 클래스를 사용하여
# 수집할 뉴스 URL 리스트를 작성하고 해당 URL의 뉴스 제목을 스크래핑 하여 URL과 제목 목록을 출력하는
# 프로그램을 작성하시오.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url_list=[]
title_list=[]

edaily_url = "https://www.edaily.co.kr/"
new_edaily_url = "https://www.edaily.co.kr/article/stock"

response = requests.get(new_edaily_url)
soup = BeautifulSoup(response.text,"html.parser")

newsList=soup.find_all('div', class_="newsbox_04")

# 뉴스 url
#a 태그 내(접근연산자 활용) href속성 (get("html 태그 속성명"))
for news in newsList:
    # print(edaily_url + news.a.get("href"))
    url_list.append(edaily_url + news.a.get("href")) 

# 뉴스 제목
for url in url_list:
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")

    title = soup.find("div",class_="news_titles")
    title_list.append(title.h1.text)

data = {"뉴스 URL":url_list,"뉴스 제목":title_list}
df=pd.DataFrame(data)

path = r".\python\ch22\task\news_edaily.csv"
df.to_csv(path,index=False)
# print(df)