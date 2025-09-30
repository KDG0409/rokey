# 리스트 연습
A = [ 1 , "abc" , True ]

# 리스트 생성
ca = [ 10 , 11 , 21 , '프로그래밍' ]
print(ca)
print( ca[0] )
print( ca[1] )
print( ca[2] )
print( ca[3] )

#리스트 변경
ca[0] = 30 # 리스트 값 재할당
print(ca)
ca[1] = 40
print(ca)

# 리스트 추가
ca.append(300) # 특정 값 추가
print(ca)
ca.append('파이썬')
print(ca)

ca.insert( 0 , 1) # 특정 인덱스 순서에 추가
print(ca)
ca.insert( 1 , 2 )
print(ca)

ca.extend( [3 , 4 , 5] ) # 다중 추가
print(ca)
ca.extend( [7 , 8 , 9] )
print(ca)

# 리스트 제거
ca.remove(9) # 특정값 제거 
print(ca)
ca.remove(8)
print(ca)

del ca[0] # 특정 인덱스 값 제거
print(ca)
del ca[0]
print(ca)
del ca[0:3]
print(ca)

ca.pop(0)
print(ca)

# 리스트 슬라이스
print(ca[1:4]) # 1번부터 3번 인덱스
print(ca[4:7]) # 4번부터 6번 인덱스