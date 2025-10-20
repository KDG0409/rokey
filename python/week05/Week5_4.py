import os
import shutil
import json

source = "./python/week05/config.ini"
if os.path.exists(source):
    path = source
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            print(line,end="")
else:
    path = source
    text = "./python/week05/config.txt"
    with open(path,'w',encoding='utf-8') as f:
        f.write("1반 = \n")
        f.write("2반 = \n")
        f.write("3반 = \n")
        f.write("4반 = ")

    with open(text,'r',encoding='utf-8') as f , open(path,'r',encoding='utf-8') as fr:
        lines = f.readlines()
        name_list=[]
        for line in lines:
            name_list.append(line)
        rlines = fr.readlines()
        class_list=[]
        for rline in rlines:
            class_list.append(rline.strip())
    
    with open(path,'w',encoding='utf-8') as f:
        data_list={}
        class_name = []
        for i in range(len(class_list)):
            f.write(class_list[i])
            f.write(name_list[i])
            data_list[class_list[i][:1]]=name_list[i][:3]
    with open("./python/week05/config.json", "w",encoding='UTF-8') as f:
        json.dump(data_list, f,ensure_ascii=False, indent=4)
    with open("./python/week05/config.json","r",encoding='UTF-8') as f:
        loaded_data = json.load(f)     
        print(loaded_data) 

    


