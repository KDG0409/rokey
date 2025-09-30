import turtle as t


#윈도우 객체
t.title("거북이 그래픽스")
t.setup(410,310)
t.bgcolor("beige")

#클래스 객체
ct=t.Turtle()
#설정
ct.shape('turtle')
ct.color('black','green')
ct.pencolor("blue")
ct.pensize(3)
#이동
ct.write("문자열",font=("arial",30,"bold")) #우측에 작성
ct.right(90)
ct.bk(100)
ct.left(90)
ct.fd(100)
ct.penup()
ct.home()
ct.pendown()
#초기화
t.reset()

#메인 루프 
t.mainloop()
t.exitonclick()