#출력 함수 정의

def welcome() :
    print('이상한 나라에 오신 것을 환영합니다.')

welcome()

def welcome(name) :
    print( name, '님 이상한 나라에 오신 것을 환영합니다.' )

welcome('엘리스')
welcome('도도새')

def draw_stars(num):
    print('*' * num) # 문자열에 곱연산 사용가능

draw_stars(3)
draw_stars(2)
draw_stars(1)