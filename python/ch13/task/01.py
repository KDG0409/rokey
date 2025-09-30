# 예외 처리 
while True:
    try:
        x=int(input("please enter a number:"))
        break
    except Exception as e:
        print(e)
        print("code operating")

# try:
#     f=open('myfile.txt')
#     s=f.readline()
#     i=int(s.srtip())
# except OSError as err:
#     print("1",err)
# except ValueError as err:
#     print("2",err)
# except Exception as err:
#     print("3",err)

# try:
#     raise NameError('HiThere')
# except NameError:
#     print('exception')