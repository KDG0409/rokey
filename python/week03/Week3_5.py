while True :
    try:
        num = int(input("숫자를 입력하세요. :"))
        print(num*num)
        break
    except ValueError:
        print("올바른 숫자를 입력하세요.")