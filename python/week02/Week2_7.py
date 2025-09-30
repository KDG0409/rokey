class Person:
    def __init__(self,name):
        self.name = name

    def introduce(self):
        return f"My name is {self.name}"
    
class Student(Person):
    def __init__(self,name,student_id):
        super().__init__(name)
        self.student_id = student_id

me = Student("김동국","06118")
print(me.introduce())
print(me.name, me.student_id)