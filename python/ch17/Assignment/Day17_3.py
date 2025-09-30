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

def bfs_queue(n,m,map):
    queue=[(0,0)] #탐색 시작 노드 삽입
    visited = []
    while queue:
        x,y=queue.pop(0)
        if x<0 or x>=n or y<0 or y>=m:
            continue
        if map[x][y] == 0 : 
            map[x][y] = 1 
            visited.append((x,y))
            queue.append(((x-1),y)) 
            queue.append(((x+1),y)) 
            queue.append((x,(y-1)))
            queue.append((x,(y+1))) 
    return visited

print(bfs_queue(n,m,map))