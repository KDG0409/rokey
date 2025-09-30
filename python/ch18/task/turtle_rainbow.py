import turtle as t

def drawRainbow(r,x,y):
    ct = t.Turtle()
    ct.shape("turtle")
    ct.pensize(5)
    ct.speed(0)

    colors = ['red','orange','yellow','green','blue','navy','purple']

    for color in colors:
        ct.color(color)
        ct.penup()
        ct.goto(x,y)
        ct.pendown()
        ct.setheading(90)
        ct.circle(r,180) 
        x += 7 # x좌표 7증가
        r += 7 # 반지름 7증가

    ct.hideturtle()

if __name__ == "__main__":
    t.title("무지개 그리기")
    t.setup(700,700)
    t.bgcolor("skyblue")
    drawRainbow(50,50,0)
    t.exitonclick()