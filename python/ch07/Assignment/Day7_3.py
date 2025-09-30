num_list = [10, 20, 30, 40, 50]

def calculate_average(nlist):
    l=len(nlist)
    sum = 0
    for num in nlist:
        sum += num
    return sum/l

average = calculate_average(num_list)
print("평균: ", average)
