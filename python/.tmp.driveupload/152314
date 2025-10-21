#1. b
#2. a
#3. b
#4. b
#5. b
#6. 
class Stack:
    def __init__(self):
        self.stack=[]
    def is_empty(self):
        if len(self.stack) == 0 :
            return True
    def push(self,data):
        self.stack.append(data)
        return self.stack
    def pop(self):
        if not self.is_empty():
            self.stack.pop()
            return self.stack
        return -1
    def top(self):
        if not self.is_empty():
            self.stack.pop()
            return self.stack[-1]
        return -1
##7. 
def postN(data):
    stack=[]
    for char in data:
        if not char in "+-*/":
            stack.append(int(char))
        else:
            b = stack.pop()
            a = stack.pop()
            if char == '+':
                stack.append(a+b)
            if char == '-':
                stack.append(a-b)
            if char == '*':
                stack.append(a*b)
            if char == '/':
                stack.append(a/b)
    return stack.pop()

print(postN('34+6*'))

#8.
class Queue:
    def __init__(self):
        self.queue=[]
    def is_empty(self):
        if len(self.queue) == 0 :
            return True
    def enqueue(self,data):
        self.queue.append(data)
        return self.queue
    def dequeue(self):
        if not self.is_empty():
            self.queue.pop(0)
            return self.queue
        return -1
    def front(self):
        if not self.is_empty():
            self.queue.pop()
            return self.queue[0]
        return -1
    
#9.
class Queue:
    def __init__(self,count):
        self.queue=[]
        self.count=count
    def is_empty(self):
        if len(self.queue) == 0 :
            return True
    def is_full(self):
        if len(self.queue) == self.count :
            return True    
    def enqueue(self,data):
        if not self.is_full():
            self.queue.append(data)
            return self.queue
        return self.queue
    def dequeue(self):
        if not self.is_empty():
            self.queue.pop(0)
            return self.queue
        return -1
    def front(self):
        if not self.is_empty():
            self.queue.pop()
            return self.queue[0]
        return -1
    
#10.
class Deque:
    def __init__(self):
        self.deque =[]
    def push_front(self,x):
        self.deque.insert(0,x)
        return self.deque
    def push_back(self,x):
        self.deque.append(x)
        return self.deque
    def pop_front(self):
        self.deque.pop(0)
        return self.deque
    def pop_back(self):
        self.deque.pop()
        return self.deque