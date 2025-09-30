# 클래스 정의, 객체생성, 클래스멤버호출(2), 인스턴스멤버호출(1)

class Singer:
    name = '아이유'

    def sing(self,str) :
        self.test = str
        print("안녕하세요.")

print(Singer.name)

iu = Singer()
print(iu.name)
#print(iu.test)
print(Singer.name)

iu.sing('hello')
print(iu.name)
print(iu.test)