class Animal:
    def speak(self):
        return "Animal speaks"
    
class Dog(Animal):
    def speak(self):
        return "Woof!"
    
name = Dog()
ans = name.speak()
print(ans)

