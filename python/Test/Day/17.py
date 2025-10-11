##1. b
##2. b
#3. b
#4. b
#5. a
#6/7.
# graph = [
#     [0,1,1,1,1,1],
#     [0,1,0,0,0,1],
#     [0,1,0,1,0,1],
#     [0,1,0,1,0,0],
#     [0,0,0,1,1,0],
#     [1,1,1,1,1,0]
# ]
# for line in graph:
#     print(line)
##8. 
# def my_map(list,n,m):
#     ans=[]
#     for i in range(n):
#         row = []
#         for j in range(m):
#             row.append(list.pop(0))
#         ans.append(row)
#     return ans

# data = [0,1,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,
#         0,1,0,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0]

# ans = my_map(data,6,6)
# print(ans)

##9.
# map = [
#     [0,1,1,1,1,1],
#     [0,1,0,0,0,1],
#     [0,1,0,1,0,1],
#     [0,1,0,1,0,0],
#     [0,0,0,1,1,0],
#     [1,1,1,1,1,0]
# ]

# n = len(map)
# m = len(map[0])

# def bfs_queue(n,m,map):
#     queue=[(0,0)] #탐색 시작 노드 삽입
#     visited = []
#     while queue:
#         x,y=queue.pop(0)
#         if x<0 or x>=n or y<0 or y>=m:
#             continue
#         if map[x][y] == 0 : 
#             map[x][y] = 1 
#             visited.append((x,y))
#             queue.append(((x-1),y)) 
#             queue.append(((x+1),y)) 
#             queue.append((x,(y-1)))
#             queue.append((x,(y+1))) 
#     return visited

# print(bfs_queue(n,m,map))

##10.
map = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,1,0,1,0,1],
    [0,1,0,1,0,0],
    [0,0,0,1,1,0],
    [1,1,1,1,1,0]
]

n = len(map)
m = len(map[0])

def dfs_stack(n,m,map):
    stack=[(0,0)] #탐색 시작 노드 삽입
    visited = []
    while stack:
        x,y=stack.pop()
        if x<0 or x>=n or y<0 or y>=m:
            continue
        if map[x][y] == 0 : 
            map[x][y] = 1 
            visited.append((x,y))
            stack.append(((x-1),y)) 
            stack.append(((x+1),y)) 
            stack.append((x,(y-1)))
            stack.append((x,(y+1))) 
    return visited

print(dfs_stack(n,m,map))