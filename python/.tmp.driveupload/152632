# Json 

import json

json_string = '{"name": "Alice", "age": 25, "city": "Seoul"}'

data = json.loads(json_string)
print(data["name"])

python_dict = {"name": "Bob", "age": 30, "city": "Busan"}

json_output = json.dumps(python_dict, indent=4)
print(json_output)

with open(r".\python\ch23\task\data.json", "w") as file:
    json.dump(python_dict, file, indent=4)

with open(r".\python\ch23\task\data.json", "r") as file:
    loaded_data = json.load(file)
print(loaded_data)    