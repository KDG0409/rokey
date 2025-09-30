# 딕셔너리 연습
a = { '이름' : '김동국' , '나이' : 24 , '키' : 178 }
print(a)
print(a['이름']) # 인덱스 사용

# 딕셔너리 추가
a['직업'] = '학생' # 인덱스 사용
print(a)

# 딕셔너리 접근
print( a['나이'] )
print( a['직업'] )
print( a.get('나이') )
print( a.get('직업') )

#딕셔너리 메소드
print( a.items() )
print( a.keys() )
print( a.values() )

#딕셔너리 삭제
del a['나이']
print(a)

a.pop('직업')
print(a)