import turtle as t

t.title("무지개색 원 그리기")
t.setup(700,700)

ct = t.Turtle()
ct.shape("turtle")
ct.pensize(2)
ct.speed(0)

colors = ['red','orange','yellow','green','blue','navy','purple']

def drawCircle(d,colors):
    for color in colors :
        ct.circle(d)
        ct.pencolor(color)
        d += 5

drawCircle(50,colors)


t.exitonclick()