# 회전 큐 구현

from collections import deque

def rotate_queue(queue,k):
    dq = deque(queue)

    dq.rotate(-k) # 음수: 왼쪽으로 k번 회전
    #for num in range(k):
    #    dq.append(dq.popleft())

    return list(dq)

queue = [1,2,3,4,5]
k = 2
print(rotate_queue(queue,k))