# 자료형 변환 프로그램
a = input ( "첫 번째 실수를 입력하세요." )
b = input ( "두 번째 실수를 입력하세요." )
fa = float ( a ) # 자료형 변환
fb = float ( b ) # 자료형 변환
print( "첫 번째 실수의 자료형은" , type(fa) , "입니다." )
print( "두 번째 실수의 자료형은" , type(fb) , "입니다." )
result = fa + fb
print(a , "+" , b , "=" , result)
result = fa - fb
print(a , "-" , b , "=" , result)