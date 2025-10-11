# 이해하기

#8.
r = int(input("세로(행): "))
c = int(input("가로(열): "))

my_map = []
for _ in range(r):
    row = list(map(int, input().split())) #각 행은 공백으로 구분된 c개의 정수를 입력받아 리스트로 변환
    my_map.append(row)

for row in my_map:
    print(row)

#9. dfs:모든 경로
visited = [[False] * c for _ in range(r)] # 모두 False로 기본 설정 (rxc배열)
dy = (-1, 1, 0, 0) #상하좌우
dx = (0, 0, -1, 1)
def dfs(y, x):
    print(y, x)
    visited[y][x] = True # 방문 시 True
    for i in range(len(dx)):
        ny = y + dy[i] #상하좌우
        nx = x + dx[i]
        if not(0 <= ny < r and 0 <= nx < c): #범위외 제외
            continue
        if my_map[ny][nx] == 1: #벽 제외
            continue
        if not visited[ny][nx]: #방문하지 않을 때
            dfs(ny, nx) #주변 길 탐색

dfs(0, 0) #시작점 설정

#10. bfs:최단경로
from collections import deque
def bfs(y, x):
    visited = [[False] * c for _ in range(r)] # 모두 False로 기본 설정 (rxc배열)
    dy = (-1, 1, 0, 0) #상하좌우
    dx = (0, 0, -1, 1)

    visited[y][x] = True # 방문 시 True
    que = deque()
    que.append((y, x)) #시작 노드 추가

    while que:
        y, x = que.popleft() #한 노드를 꺼냄
        print(y, x)
        for i in range(len(dx)):
            ny = y + dy[i] #상하좌우
            nx = x + dx[i]
            if not(0 <= ny < r and 0 <= nx < c): #범위외 제외
                continue
            if my_map[ny][nx] == 1: #벽 제외
                continue
            if not visited[ny][nx]: #방문하지 않을 때
                visited[ny][nx] = True #방문처리하고
                que.append((ny, nx)) #주변 노드 탐색

bfs(0, 0) #시작점 설정

