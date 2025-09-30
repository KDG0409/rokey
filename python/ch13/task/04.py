#split()

my_string = "Python is a popular programing language"
split_list = my_string.split()
print(split_list)

#strip()

my_string = "    Python is a popular programing language    "
strip_list = my_string.strip()
print(strip_list)

#join()

print("-".join(split_list))