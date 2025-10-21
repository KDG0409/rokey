# generator1.py
import time
import re

def longtime_job():
    print("job start")
    time.sleep(1) # 1초 지연
    return "done"

#리스트 컴프리헨션
list_job = [longtime_job() for i in range(5)]
print(list_job[0])
print(type(list_job))

list_job = iter(list_job)
print(type(list_job))
print(next(list_job))

#제너레이터 컴프리헨션
list_job2 = (longtime_job() for i in range(5))
print(next(list_job2))
print(type(list_job2))

bad_words = ["spam","ad","click"]
pattern = "|".join([re.escape(word) for word in bad_words]) #"spam|ad|click"
print(type(pattern)) #str
text = "This is spam ad"
m = re.findall(pattern,text) #['spam', 'ad']
print(m)