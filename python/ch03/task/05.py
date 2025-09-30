# if-elif-else 조건문

tscore = 920

if  tscore > 990 :
    print( "당신의 토익 점수는" , tscore , "잘못되었습니다.")
elif tscore >= 900 :
    print( " 당신의 토익 점수는" , tscore , "상위권 점수입니다." )    
elif tscore >= 700 :
    print( " 당신의 토익 점수는" , tscore , "중상위권 점수입니다." )
elif tscore >= 500 :
    print( " 당신의 토익 점수는" , tscore , "중위권 점수입니다." )
elif tscore >= 0 :
    print( " 당신의 토익 점수는" , tscore , "하위권 점수입니다." )
else :
    print( " 당신의 토익 점수는" , tscore , "잘못되었습니다.")
print( " if문 종료됨 ")