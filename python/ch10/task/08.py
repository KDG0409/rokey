# class 오버라이딩

class Animal:
    def eat(self):
        print('먹다')
    def move(self):
        print('이동하다')
    def sound(self):
        print('소리내다')

class Cat(Animal):
    def eat(self):
        print('잡식하다')
    def sound(self):
        print('meww')
        super().sound() #super 클래스 : 부모 클래스 메소드 호출

cat1 = Cat()
cat1.eat()
cat1.move()
cat1.sound()  