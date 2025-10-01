graph = {
    '두':['산','로','키'],'산':['두','부'],'부':['산'],
    '로':['두'],'키':['두','트'],'트':['키','캠','프'],
    '캠':['트'],'프':['트']
}

def my_dfs(graph,snode):
    stack = []
    visited = []
    stack.append(snode)
    while stack:
        node = stack.pop()
        if node not in visited :
            stack.extend(reversed(graph[node]))
            visited.append(node)
    return visited

print(my_dfs(graph,'두'))   

def my_bfs(graph,snode):
    queue = []
    visited = []
    queue.append(snode)
    while queue:
        node = queue.pop(0)
        if node not in visited :
            queue.extend(graph[node])
            visited.append(node)
    return visited

print(my_bfs(graph,'두'))   

