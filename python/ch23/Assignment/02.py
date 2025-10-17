import requests
import json

url='https://jsonplaceholder.typicode.com/todos/1'
r = requests.get(url)
data = json.loads(r.text)
print(data)