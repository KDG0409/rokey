import re

data = '^[0-9][abc]*'
data2 = 'a.b'
data3 = '^hello'
data4 = 'hello$'
data5 = 'hello'

p = re.compile(data)
m = p.search('3banana')
print(m)

p2 = re.compile(data2)
m2 = p2.search('a\rb')
print(m2)

p3 = re.compile(data3)
m3 = p3.search('kenneth,hello')
print(m3)
m3 = p3.search('hello,kenneth')
print(m3)

p4 = re.compile(data4)
m4 = p4.search('kenneth,hello,kim,hello')
print(m4)
m4 = p4.search('hello,kenneth,hello,kim')
print(m4)

p5 = re.compile(data5)
m5 = p5.search('kenneth,hello,kim,hello')
print(m5)
m5 = p5.search('hello,kenneth,hello,kim')
print(m5)

p6 =re.compile("[a-z]+")
print(p6.search("3pyt8hon"))

p6 =re.compile("[a-z]*")
print(p6.search("3pyt8hon"))

p6 =re.compile("[a-z]*")
print(p6.findall("3pyt897hon"))

p6 =re.compile("ca*t")
print(p6.findall("cat"))