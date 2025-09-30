# 1부터 입력받은 정수까지의 총합을 구하는 프로그램을 while문으로 작성

num = int(input("총합을 구하려는 수를 입력하세요."))

i = 1
total = 0
while i <= num :
    total += i
    i += 1

print( "1부터" , num , "까지의 총합은" , total , "이다." )