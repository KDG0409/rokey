# class 상속 , super 클래스

class Human:
    eyes = 2
    nose = 1
    mouth = 1

    def __init__(self,name,age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"이름:{self.name}")
        print(f"나이:{self.age}")
    
    def eat(self):
        print('먹다')

    def sleep(self):
        print('자다')

    def talk(self):
        print('말하다')

class Student(Human): 
    def __init__(self,name,age,studentNum):
        super().__init__(name,age)
        self.studentNum = studentNum

    def introduce(self):
        super().introduce()
        print(f"학번:{self.studentNum}")


    def study(self):
        print('공부하다')

Hong = Human("홍길동",33)
Hong.introduce()
kim = Student("김동국",25,2020112327)
kim.introduce()
kim.study()