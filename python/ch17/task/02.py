fgraph = {1:[2,3,4],2:[1,5],3:[1,6],4:[1,6],5:[2,6],6:[3,4,5]}

def my_dfs(graph,snode):
    stack = []
    visited = []
    stack.append(snode)
    while stack:
        node = stack.pop()
        if node not in visited:
            stack.extend(reversed(graph[node]))
            visited.append(node)
    return visited

def my_bfs(graph,snode):
    queue = []
    visited = []
    queue.append(snode)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            queue.extend(graph[node])
            visited.append(node)
    return visited

print(my_dfs(fgraph,1))
print(my_bfs(fgraph,1))