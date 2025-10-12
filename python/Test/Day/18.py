#1. b
#2. c
#3. a
#4. c
#5. b
#6/7/8/9/10. 
import turtle as t
t.setup(800,600)
t.bgcolor("black")
def drawStar(x,y,d,color):
    cts=t.Turtle()
    cts.speed(5)
    cts.pensize(1)
    cts.penup()
    cts.goto(x,y)
    cts.pendown()
    cts.color(color)
    for i in range(6):
        cts.left(144)
        cts.fd(d)
    cts.hideturtle()

def drawMoon(x,y,r,color):
    ctm=t.Turtle()
    ctm.speed(5)
    ctm.pensize(1)
    ctm.penup()
    ctm.goto(x,y)
    ctm.pendown()
    ctm.color(color)   
    ctm.circle(r)
    ctm.hideturtle() 

for i in range(6):
    drawStar(300-100*i,100,30,"gold")
drawMoon(0,200,100,"white")

t.exitonclick()