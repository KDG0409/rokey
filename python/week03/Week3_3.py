import re

data = """python one
life is too short
python two
you need python
python three"""

m = re.findall(r"python[ \w]+",data)
print(m)