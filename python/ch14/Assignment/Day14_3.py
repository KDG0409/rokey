import re

p = re.compile(r"\d+-\d+-\d+")
text = "연락처: 010-1234-5678, 02-987-6543, 031-456-7890"
m= p.findall(text)
print(m)