# 이중 for 문을 사용하여 1단부터 5단 까지 구구단을 출력하는 프로그램을 작성

ans = 0

for i in range( 1 , 6 ) :
    for num in range( 1 , 10 ):
        ans = num * i
        print( i , "*" , num , "=" , ans)