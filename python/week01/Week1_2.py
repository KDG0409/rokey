score = int(input("점수를 입력하세요 : "))

if score < 0 :
    print("점수가 잘못 입력되었습니다.")
elif score >= 90 :
    print("점수 :" , score , "," , "학점 :" , "A학점")
elif score >= 80 :
    print("점수 :" , score , "," , "학점 :" , "B학점")
elif score >= 70 :
    print("점수 :" , score , "," , "학점 :" , "C학점")
else :
    print("점수 :" , score , "," , "학점 :" , "F학점")