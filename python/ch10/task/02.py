# import 모듈명

import m01 as m

p2 = m.Cvalue() #클래스 정의
p2.add(11)
p2.add(21)
p2.add(31)
p2.fprint()

value = m.plus(10,20)
print(value)

# 기반 모듈 내 생성된 객체 사용

m.p1.add("핸드폰")
value2 = m.p1.lista
print(value2)