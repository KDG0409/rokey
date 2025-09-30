# if-elif-else 조건문 2

text = "월요일 부터 일요일 까지 영어로 번역하고 싶은 요일을 입력하세요. : "
A = input(text)
if A == "월요일" :
    print( A , "은 영어로 Monday 입니다." )
elif A == "화요일" :
    print( A , "은 영어로 Tuesday 입니다." )    
elif A == "수요일" :
    print( A , "은 영어로 Wednesday 입니다." )
elif A == "목요일" :
    print( A , "은 영어로 Thursday 입니다." )
elif A == "금요일" :
    print( A , "은 영어로 Friday 입니다." )
elif A == "토요일" :
    print( A , "은 영어로 Saturday 입니다." )
elif A == "일요일" :
    print( A , "은 영어로 Sunday 입니다." )
else :
    print( " 한글 요일을 잘못 입력했습니다." )