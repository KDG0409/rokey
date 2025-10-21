#1. b
#2. b
#3. b
#4. b
#5. a
#6. 
# graph=[
#     [0,1,1,1,1,1],[0,1,0,0,0,1],[0,1,0,1,0,1],[0,1,0,1,0,0],[0,0,0,1,1,0],[1,1,1,1,1,0]
# ]
#7.
# for nodes in graph:
#     print(nodes)
#8.
# def myMap(data,n,m):
#     graph=[]
#     for i in range(n):
#         row=[]
#         for j in range(m):
#             row.append(data.pop(0))
#         graph.append(row)
#     return graph
# data = [0,1,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,
#         0,1,0,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0]
# print(myMap(data,6,6))
#9.
graph=[
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,1,0,1,0,1],
    [0,1,0,1,0,0],
    [0,0,0,1,1,0],
    [1,1,1,1,1,0]
]
n=len(graph)
m=len(graph[0])

def myDfs(x,y):
    stack=[(x,y)]
    visited=[]
    while stack:
        x,y=stack.pop()
        if x<0 or x>=n or y<0 or y>=m:
            continue
        if graph[x][y]==1:
            continue
        if graph[x][y]==0:
            graph[x][y]=1
            stack.append((x+1,y))
            stack.append((x-1,y))
            stack.append((x,y+1))
            stack.append((x,y-1))
            visited.append((x,y))
    return visited

print(myDfs(0,0))

#10 
graph=[
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,1,0,1,0,1],
    [0,1,0,1,0,0],
    [0,0,0,1,1,0],
    [1,1,1,1,1,0]
]
n=len(graph)
m=len(graph[0])

def myDfs(x,y):
    stack=[(x,y)]
    visited=[]
    while stack:
        x,y=stack.pop(0)
        if x<0 or x>=n or y<0 or y>=m:
            continue
        if graph[x][y]==1:
            continue
        if graph[x][y]==0:
            graph[x][y]=1
            stack.append((x+1,y))
            stack.append((x-1,y))
            stack.append((x,y+1))
            stack.append((x,y-1))
            visited.append((x,y))
    return visited

print(myDfs(0,0))