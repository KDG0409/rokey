#24
# num = 0

# while num <101 :
#     if num %3 != 0 :
#         num+=1
#         continue
#     print(num)
#     num+=1

#25
# num = int(input("총합을 구하려는 수를 입력하세요.:"))
# total = 0
# for i in range(1,num+1):
#     total += i
# print(f"1부터{num}까지의 총합은 {total}이다.")

##27
# num = int(input("총합을 구하려는 수를 입력하세요.:"))
# total = 0
# i = 0
# while i < num+1 :
#     total += i
#     i += 1

# print(f"1부터{num}까지의 총합은 {total}이다.")

#28
# num = int(input("총합을 구하려는 수를 입력하세요.:"))
# total = 0
# i = 0
# while True :
#     total += i
#     i += 1
#     if i > num :
#         break

# print(f"1부터{num}까지의 총합은 {total}이다.")

##29
# num1 = int(input("계산할 첫번째 수를 입력한다.:"))
# num2 = int(input("계산할 두번째 수를 입력한다.:"))
# total = 0
# for num in range(num1,num2+1):
#     total += num
# print(total)

# total = 0
# i=num1
# while num2 >= i :
#     total += i 
#     i += 1
# print(total)

#30
# num1 = int(input("계산할 첫번째 수를 입력한다.:"))
# num2 = int(input("계산할 두번째 수를 입력한다.:"))
# total = 0
# for num in range(num1,num2+1):
#     if num % 3 == 0 :
#         total += num
# print(total)

# total = 0
# i=num1
# while num2 >= i :
#     if i %3 == 0 :
#         total += i 
#         i += 1
#     i += 1
# print(total)

##31
# Myname =''
# while True:
#     print("이름을 입력하세요.:")
#     Myname = input()
#     if Myname == 'hongkildong' :
#         continue
#     else:
#         print('패스워드를 입력하세요.')
#         MyPass = input()
#         if MyPass == "hahaha":
#             break
# print('확인되었습니다.')

#33
# for i in range(1,10):
#     print(f"1*{i}={i}")

#34
# for i in range(1,10):
#     print(f"2*{i}={2*i}")

#35
# for i in range(1,6):
#     for j in range(1,10):
#         print(f"{i}*{j}={i*j}")

##37
# for i in range(1,6):
#     for j in range(1,i+1):
#         print(j,end = " ")
#     print("")

# lista = ['A','B','C','D','E']
# for i in range(1,6):
#     for j in range(0,i):
#         print(lista[j],end = " ")
#     print("")