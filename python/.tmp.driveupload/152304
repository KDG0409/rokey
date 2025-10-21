# 괄호 짝 검사
def checkChar(characters):
    stack = []
    for char in characters:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack:
                return False
            data = stack.pop()
            if data == '[' and char !=']':
                return False
            elif data == '(' and char !=')':
                return False
            elif data == '{' and char !='}':
                return False
    return len(stack) == 0

print(checkChar('[]'))

# 회전 큐 구현
from collections import deque

def rotate_queue(queue,k):
    dq = deque(queue)
    for num in range(k):
        dq.append(dq.popleft())
    return dq

# dq.rotate(-k) 왼쪽으로 k번 회전

queue = [1,2,3,4,5]
k = 2
print(rotate_queue(queue,k))