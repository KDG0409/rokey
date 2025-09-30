import re

p = re.compile(r"\b[A-Z]+\b")
text = "NASA is working on AI projects with IBM and Google."
m = p.findall(text)
print(m)