#1. b
#2. c
#3. a
#4. c
#5. b
#6. 
import turtle as t
t.title("")
t.setup(800,600)
t.bgcolor('black')

def drawStar(x,y,d,color):
    ct=t.Turtle()
    ct.color(color)
    ct.shape('turtle')
    ct.speed(5)
    ct.pensize(1)
    ct.penup()
    ct.goto(x,y)
    ct.pendown()
    for i in range(5):
        ct.fd(d)
        ct.right(144)
    ct.hideturtle()

def drawMoon(x,y,r,color):
    ct=t.Turtle()
    ct.color(color)
    ct.shape('turtle')
    ct.speed(0)
    ct.penup()
    ct.goto(x,y)
    ct.pendown()
    ct.circle(r)
    ct.hideturtle()

drawMoon(0,200,100,'white')
for i in range(5):
    drawStar(50*i-100,0,20,'brown')

t.exitonclick()
