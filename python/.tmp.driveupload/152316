#DFS
# def myDfs(graph,start_node):
#     stack=[]
#     visited=[]
#     stack.append(start_node)
#     while stack:
#         node=stack.pop()
#         if node not in visited:
#             stack.append(reversed(graph(node)))
#             visited.append(node)
#     return visited
#BFS
# def myBfs(graph,start_node):
#     stack=[]
#     visited=[]
#     stack.append(start_node)
#     while stack:
#         node=stack.pop(0)
#         if node not in visited:
#             stack.append(graph(node))
#             visited.append(node)
#     return visited
#얼음찾기문제
graph = [
    [0,0,1,0,1],
    [1,0,1,0,0],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [0,0,0,1,1]
]
n=len(graph)
m=len(graph[0])

def find_ice(x,y):
    stack=[(x,y)]
   # visited=[]
    while stack:
        x,y=stack.pop()
        if x<0 or y<0 or x>=n or y>=m :
            continue
        if graph[x][y]==0:
            graph[x][y]=1
            stack.append((x+1,y))
            stack.append((x-1,y))
            stack.append((x,y+1))
            stack.append((x,y-1))
            print(x,y)
            #visited.append((x,y))
    return True #visited

result=0
for i in range(n):
    for j in range(m):
        if graph[i][j]==0:
            if find_ice(i,j)==True:
                result+=1
print(result)