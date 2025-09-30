import turtle as t

t.title("정사각형 그리기")
t.setup(700,700)

ct = t.Turtle()
ct.shape("turtle")
ct.pencolor("skyblue")
ct.pensize(5)

ct.fd(100)
ct.right(90)
ct.fd(100)
ct.right(90)
ct.fd(100)
ct.right(90)
ct.fd(100)
ct.right(90)

ct.home()
ct.reset()


t.exitonclick()