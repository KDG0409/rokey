import os
print(os.getcwd())

f = open("C:/rokey/python/ch12/task/file1.txt",'r') # 내장 클래스->객체생성

# for i in range(1,11):
#     data = "%d번째 줄입니다.\n" %i
#     f.write(data) #파일 쓰기 메소드,w(기존 데이터 삭제),a(기존 데이터 아래에 추가)

# print(f.readline()) #파일 읽기 메소드,r -> 반환값:첫째 줄
# print(f.readlines()) #파일 읽기 메소드,r -> 반환값: 전체 내용, 리스트 요소
# print(f.read()) #파일 읽기 메소드,r -> 반환값: 전체 내용, 문자열로 반환


f.close() #파일 닫기 메소드