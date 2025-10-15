#1. b
#2. a
#3. b
#4. b
#5. b
#6. 
# class Stack:
#     def __init__(self):
#         self.stack=[]
#     def is_empty(self):
#         if len(self.stack)==0:
#             return True
#     def pop(self):
#         if not self.is_empty():
#             return self.stack.pop()
#         return -1
#     def push(self,data):
#         return self.stack.append(data)
#     def top(self):
#         if not self.is_empty():
#             return self.stack[-1]
#         return -1
#7.
def pn(chars):
    stack=[]
    for char in chars:
        if char not in '+-*/':
            stack.append(int(char))
        if char in '+-*/':
            data2=stack.pop()
            data1=stack.pop()
            if char =='+':
                stack.append(data1+data2)
            if char =='-':
                stack.append(data1-data2)
            if char =='*':
                stack.append(data1*data2)
            if char =='/':
                stack.append(data1/data2)
    return stack.pop()
print(pn("34+5+"))
                
#8.
# class queue:
#     def __init__(self):
#         self.queue=[]
#     def is_empty(self):
#         if len(self.queue)==0:
#             return True
#     def dequeue(self):
#         if not self.is_empty():
#             return self.queue.pop(0)
#         return -1
#     def enqueue(self,data):
#         return self.queue.append(data)
#     def front(self):
#         if not self.is_empty():
#             return self.self.queue[0]
#         return -1
#9.
# class queue:
#     def __init__(self,len):
#         self.queue=[]
#         self.len=len
#     def is_empty(self):
#         if len(self.queue)==0:
#             return True
#     def is_full(self,len):
#         if len(self.queue) >= self.len:
#             return True        
#     def dequeue(self):
#         if not self.is_empty():
#             return self.queue.pop(0)
#         return self.queue
#     def enqueue(self,data):
#         if not self.is_full():
#             return self.queue.append(data)
#         return self.queue
#     def front(self):
#         if not self.is_empty():
#             return self.self.queue[0]
#         return -1
#10.
# class deque:
#     def __init__(self):
#         self.deque=[]
#     def is_empty(self):
#         if len(self.deque)==0:
#             return True
#     def pop_front(self):
#         if not self.is_empty():
#             return self.deque.pop(0)
#         return -1
#     def pop_back(self):
#         if not self.is_empty():
#             return self.deque.pop()
#         return -1
#     def push_front(self,data):
#         return self.deque.insert(0,data)
#     def push_back(self,data):
#         return self.deque.append(data)
