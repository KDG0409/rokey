#1. Radiobutton
##2. 1 messagebox 2 title 3 on_button_click
#3. 1 w 2 write 3 r
#4. 
# with open("data.txt",'a',encoding='UTF-8')as file:
#     file.write("11번째 줄 입니다.")
#5.
# while True:
#     try:
#         num = int(input('숫자를 입력하세요.:'))
#         print(num**2)
#         break
#     except Exception:
#         print("올바른 숫자를 입력하세요.")
##6.
# result = map(lambda x : x**2 ,[10,20,30,40,50])
# print(list(result))
##7.
# import re
# data = """python one
# life is too short
# python two
# you need python
# python three"""
# m = re.findall(r'^python[\w ]+',data,re.M)
# print(m)
#8. 
# import re
# data = input('이메일을 입력하세요.:')
# m = re.findall(r'\w+@\w+.\w+',data)
# if not m :
#     print('다시입력하세요.')
#9. 1 position 2 StopIteration 3 ListIterator
#10.
# def my_gen(n):
#     for i in range(1,n+1):
#         yield i**2
# ans = my_gen(10)    
# for num in ans:
#     print(num)
