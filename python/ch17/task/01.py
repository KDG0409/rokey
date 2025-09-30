# 그래프 표현

my_graph = { "A":["B","C","D"],"B":["A","E"],"C":["A","F","G"],"D":["A","H"],\
            "E":["B","I"],"F":["C","G"],"I":["E"],"G":["C"],"H":["D"],"J":["F"]}

#DFS함수
def my_dfs(graph,start_node):
    stack = list()
    visited = list()
    stack.append(start_node)
    while stack:
        node = stack.pop()
        if node not in visited:
            stack.extend(reversed(graph[node]))
            visited.append(node)
    print(visited)
    return visited

my_dfs(my_graph,"A")      

#BFS함수
def my_bfs(graph,start_node):
    queue = list()
    visited = list()
    queue.append(start_node)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            queue.extend(graph[node])
            visited.append(node)
    print(visited)
    return visited

my_bfs(my_graph,"A")    
