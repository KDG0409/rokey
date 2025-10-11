#DFS
def my_dfs(graph,start):
    stack = []
    visited = []
    stack.append(start)
    while stack:
        node = stack.pop()
        if node not in visited:
            stack.extend(reversed(graph[node]))
            visited.append(node)
    return visited
#BFS
def my_bfs(graph,start):
    queue = []
    visited = []
    queue.append(start)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            queue.extend(graph[node])
            visited.append(node)
    return visited

fgraph = {1:[2,3,4],2:[1,5],3:[1,6],4:[1,6],5:[2,6],6:[3,4,5]}
print(my_dfs(fgraph,1))
print(my_bfs(fgraph,1))    

# 얼음 수 찾기 이해 (출제x)
graph = [
    [0,0,1,0,1],
    [1,0,1,0,0],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [0,0,0,1,1]
]
n = len(graph)
m = len(graph[0])

def find_ice(graph,start):
    stack=[(x,y)]
    while stack:
        x,y=stack.pop()
        if x<0 or x>=n or y<0 or y>=m: 
            continue  
        if graph[x][y] == 0 : 
            graph[x][y] = 1 
            stack.append(((x-1),y)) 
            stack.append(((x+1),y)) 
            stack.append((x,(y-1))) 
            stack.append((x,(y+1)))
    return True   

for i in range(n):
    for j in range(m):
        if graph[i][j] == 0 :
            if find_ice(i,j) == True:
                result += 1

print(result)     