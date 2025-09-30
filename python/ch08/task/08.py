# 선택 정렬 알고리즘_오름차순 정렬 (함수x)

ca=[21,10,11,15,13]

min = ca[0]
emin = 0
for num in range(1,5,1) : #인덱스 번호 이용
    if min > ca[num] :
        min = ca[num]
        emin = num
temp = ca[0]   # 최소값 0번 인덱스로 스왑
ca[0] = ca[emin]
ca[emin] = temp

min = ca[1]
emin = 1
for num in range(2,5,1) : #인덱스 번호 이용
    if min > ca[num] :
        min = ca[num]
        emin = num
temp = ca[1]   # 최소값 1번 인덱스로 스왑
ca[1] = ca[emin]
ca[emin] = temp

min = ca[2]
emin = 2
for num in range(3,5,1) : #인덱스 번호 이용
    if min > ca[num] :
        min = ca[num]
        emin = num
temp = ca[2]   # 최소값 2번 인덱스로 스왑
ca[2] = ca[emin]
ca[emin] = temp

min = ca[3]
emin = 3
for num in range(4,5,1) : #인덱스 번호 이용
    if min > ca[num] :
        min = ca[num]
        emin = num
temp = ca[3]   # 최소값 3번 인덱스로 스왑
ca[3] = ca[emin]
ca[emin] = temp

print(ca)