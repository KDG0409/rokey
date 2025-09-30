# 10% 할인 함수 정의
def print_lower_price() :
    cost = 0
    cost = int(input("현재 가격을 입력하세요.:"))
    new_cost = int(cost * 0.9)
    print("할인 된 가격:",new_cost)

print_lower_price()