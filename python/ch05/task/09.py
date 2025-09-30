# 1부터 입력받은 정수까지의 총합을 구하는 프로그램을 break문으로 작성

num = int(input("총합을 구하고 싶은 정수를 입력하세요."))

i = 1
total = 0
while True :
    total = total + i 
    i = i + 1
    if num < i :
        break

print( "1부터" , num , "까지의 총합은", total , "입니다." )