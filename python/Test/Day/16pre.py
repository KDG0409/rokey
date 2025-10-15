# 괄호 짝 검사
def checkit(chars):
    stack=[]
    for char in chars:
        if char in "({[":
            stack.append(char)
        elif char in ")]}":
            if len(stack)==0:
                return False
            data=stack.pop()
            if data=='(' and char != ')':
                return False
            if data=='[' and char != ']':
                return False
            if data=='{' and char != '}':
                return False
    return len(stack) == 0

print(checkit("(안녕하세요.)"))

# 회전 큐 구현
from collections import deque
def Ro_deque(deque,k):
    for i in range(k):
        deque.append(deque.popleft())
    return deque
data=deque([1,2,3,4,5])        
print(Ro_deque(data,3))