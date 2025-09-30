numbers = [10, 20, 30]
try :
    num = int(input("인덱스를 입력하세요. : "))
    print(numbers[num])
except (IndexError,ValueError):
    print("잘못된 인덱스입니다.")