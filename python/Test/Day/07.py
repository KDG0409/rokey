## 팩토리얼 계산
# def my_fac(n):
#     if n == 1 :
#         return 1
#     else:
#         return n * my_fac(n-1)
    
# print(my_fac(5))

#1. a global
#2. b
#3. 50,100, 디폴트 매개변수가 있는 경우 입력이 없으면 기본값 사용
#4. 인수와 매개변수가 1대1로 대응되지 않아서 오류발생
#5. 15 add_10에서 outer함수 호출, outer함수가 inner함수를 호출하여 리턴값 반환
##6. result가 지역변수이므로 전역공간에서 사용 불가
#8. ACB
#9. 
# def check_odd_even(n):
#     if n %2 == 0 :
#         print('Even')
#     else:
#         print('Odd')
# check_odd_even(4)
# check_odd_even(7)
#10.
# def calculate_average(num):
#     return sum(num)/len(num)
# num_list = [10, 20, 30, 40, 50]
# average = calculate_average(num_list)
# print("평균:",average)
