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