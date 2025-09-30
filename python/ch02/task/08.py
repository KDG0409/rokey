# +연산자의 두 가지 기능

print( " 첫 번째 정수를 입력하세요. ") 
ra = input ()
print( " 두 번째 정수를 입력하세요. ") 
rb = input ()
rc = ra + rb
print( "문자열인 경우" , ra , "+" , rb , "의 값은" , rc , "이다." )
ra = int(ra)
rb = int(rb)
rc = ra + rb 
print( "정수인 경우" , ra , "+" , rb , "의 값은" , rc , "이다." )