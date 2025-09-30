num1 = int(input( "숫자를 입력하세요." ))

if num1 < 0 :
    print( "잘못된 정보입니다." )
elif num1 % 2 == 0 :
    print( num1 , "은 짝수입니다." )
else :
    print( num1 , "은 홀수입니다." )