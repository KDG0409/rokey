import json

python={"name": "홍길동" , "age": 25 , "city" : "서울"}

with open(r".\python\ch23\task\data.json", "w",encoding='UTF-8') as file:
    json.dump(python, file, indent=4)