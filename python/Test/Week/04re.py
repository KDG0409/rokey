#1.
class Stack:
    def __init__(self):
        self.stack=['두산','로키','부트']
    def push(self,data):
        self.stack.append(data)
        return self.stack
ans=Stack()
print(ans.push('캠프'))
#2.
class Stack:
    def __init__(self):
        self.stack=['두산','로키','부트']
    def push(self,data):
        self.stack.append(data)
        return self.stack
    def pop(self):
        self.stack.pop()
        return self.stack
ans=Stack()
print(ans.push('캠프'))
print(ans.pop())
#3.트
#4.트
