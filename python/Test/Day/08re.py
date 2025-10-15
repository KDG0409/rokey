#1. b
#2. a
#3. 
# arr = [3, 6, 9, 12]
# arr[0],arr[2]=arr[2],arr[0]
# print(arr)
#4. a
#5. a
#6.
# a = [3, 6, 7, 4, 9, 10, 13]
# for i in range(len(a)):
#     if a[i] %2 != 0 :
#         l=i
# for j in range(len(a)-1,-1,-1):
#     if a[j] %2 == 0 :
#         f=j

# a[l],a[f]=a[f],a[l]
# print(a)
#7.
# def fmax(list):
#     max=list[0]
#     for i in range(len(list)):
#         if max<list[i]:
#             max=list[i]
#     return max

# def fmin(list):
#     min=list[0]
#     for i in range(len(list)):
#         if min>list[i]:
#             min=list[i]
#     return min
# print(fmax([2,8,3,9,4,8,3]))
# print(fmin([2,8,3,9,4,8,3]))
#8. b
#9. a
#10. 
# data={ 'a': 10, 'b': 20, 'c': 30 }
# def dsum(dict):
#     data=dict.values()
#     return sum(data)
# print(dsum(data))

# 선택정렬 알고리즘
# a = [3, 6, 7, 4, 9, 10, 13]
# def fselsort(num_list):
#     for j in range(len(num_list)):
#         min = num_list[j]
#         for i in range(j,len(num_list)):
#             if min > num_list[i]:
#                 min = num_list[i]
#                 num_list[j],num_list[i]=num_list[i],num_list[j]
#     return num_list
# print(fselsort(a))
        