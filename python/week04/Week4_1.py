class Stack:
    def __init__(self):
        self.stack = ['두산', '로키', '부트']
    def push(self,data):
        self.stack.append(data)
        return self.stack
    def is_empty(self):
        if len(self.stack) == 0 :
            return True
        return False
    def pop(self):
        if not self.is_empty():
            self.stack.pop()
            return self.stack
        return

lista = Stack()
print(lista.push('캠프'))
print(lista.pop())
