data = "안녕하세요.\n"+"안녕하세요.\n"+"안녕하세요.\n"

def write_file(file_path):
    with open(file_path,'w',encoding = "utf-8") as file:
        file.write(data)

def generate(lista):
    yield(lista[0])
    yield(lista[1])
    yield(lista[2])

write_file("C:/rokey/python/ch15/task/file1.txt")

def read_file():
    with open("C:/rokey/python/ch15/task/file1.txt",'r',encoding = "utf-8") as f:
        for line in f.readlines():
            yield line.strip()

lines = read_file()
print(type(lines))
for line in lines:
    print(line)


    #line = f.readlines()
    #gen1 = generate(line)
    #print(type(gen1))
    #print(type(line))

    #print(next(gen1))
    #print(next(gen1))
    #print(next(gen1))

    # lines = f.read().split("\n")
    # print(type(lines))
    # gen2 = (line for line in lines)
    # print(type(gen2))

    # print(next(gen2))
    # print(next(gen2))
    # print(next(gen2))
