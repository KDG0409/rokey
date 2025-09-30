i = 1
odd = []
even = []

while i < 31 :
    if i % 2 == 0 :
        even.append (i)
        i += 1
    else :
        odd.append (i)
        i += 1

print ( "홀수 : " , odd)
print ( "짝수 : " , even)