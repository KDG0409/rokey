a = [3, 6, 7, 4, 9, 10, 13]

for num in range(0,len(a)):
    if a[num] % 2 == 0 :
        num1 = num
        break

for num in range(0,len(a)):
    if a[num] % 2 != 0 :
        num2 = num

a[num1],a[num2] = a[num2],a[num1]
print(a)