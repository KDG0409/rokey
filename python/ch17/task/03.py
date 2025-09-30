# 이차원 배열 리스트 작성

# 0으로 이어진 영역의 개수를 세는 프로그램
graph = [
    [0,0,1,0,1],
    [1,0,1,0,0],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [0,0,0,1,1],
]
n = len(graph) #행 개수 (세로)
m = len(graph[0]) #열 개수 (가로)

def dfs_stack(x,y):
    stack=[(x,y)] #탐색 시작 노드 삽입

    while stack:
        x,y=stack.pop() #스택에서 노드 하나 꺼냄
        if x<0 or x>=n or y<0 or y>=m: #주어진 범위 외에서 무시
            continue

        if graph[x][y] == 0 : #방문하지 않았을 때 해당 노드 방문처리 #if node not in visited
            graph[x][y]=1 #visited.append(node)
            stack.append(((x-1),y)) #위쪽   #extend(graph[node])
            stack.append(((x+1),y)) #아래쪽
            stack.append((x,(y-1))) #왼쪽
            stack.append((x,(y+1))) #오른쪽

            # print("----------------------")
            # print(stack)
            # for i in graph:
            #     print(i)
    return True

result = 0

for i in range(n):
    for j in range(m):
        if graph[i][j] == 0 : #리스트의 얕은 복사
            # print(i,j)
            if dfs_stack(i,j) == True:
                result += 1

print(result)