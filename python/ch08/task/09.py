 # 선택 정렬 알고리즘_오름차순 정렬 (이중 for문)
ca=[21,10,11,15,13]

l=len(ca)
for i in range(0,l-1,1) : # l도 무방함
    min = ca[i]
    emin = i
    for num in range(i+1,l,1) : #인덱스 번호 이용
        if min > ca[num] :
            min = ca[num]
            emin = num
    temp = ca[i]   
    ca[i] = ca[emin]
    ca[emin] = temp

print(ca)