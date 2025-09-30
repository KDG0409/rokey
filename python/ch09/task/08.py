class Human :
    def __init__(self,age,name):
        self.age=age
        self.name=name

    def intro(self):
        print(self.age)
        print(self.name)

me = Human(25,'김동국')
me.intro()

