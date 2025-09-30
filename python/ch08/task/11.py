#선택 정렬 연습

c = [132,45,23,67,84]

def arr(ca) :
    l = len(ca)
    for i in range(0,l-1,1) :
        min = ca[i]
        emin = i
        for num in range(i,l,1):
            if min > ca[num]:
                min = ca[num]
                emin = num
        temp = ca[i]
        ca[i] = ca[emin]
        ca[emin] = temp
    return ca

c=arr(c)
print(c)
