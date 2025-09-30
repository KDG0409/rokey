import turtle as t

t.title("원그리기")
t.setup(700,700)

ct = t.Turtle()
ct.shape("turtle")
ct.pensize(3)
ct.color("black","gold")
ct.begin_fill()
ct.circle(20)
ct.end_fill()

t.exitonclick()