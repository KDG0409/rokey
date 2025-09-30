with open("C:/rokey/python/ch12/task/order.txt",'a',encoding="utf-8") as f :
    order = input("주문을 입력하세요.: ")
    f.write(order+"\n")
with open("C:/rokey/python/ch12/task/order.txt",'r',encoding="utf-8") as f :
    print(f.read())