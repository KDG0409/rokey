import turtle as t

t.setup(800,600)
t.bgcolor('black')

def drawStar(x,y,length,color):
    star = t.Turtle()
    star.pensize(1)
    star.color(color)
    star.shape('turtle')
    star.speed(5)
    star.penup()
    star.goto(x,y)
    star.pendown()
    for i in range(5):
        star.forward(length)
        star.right(144)
    star.hideturtle()

def drawMoon(x,y,d,color):
    moon = t.Turtle()
    moon.color(color,color)
    moon.begin_fill()
    moon.shape('turtle')
    moon.speed(5)
    moon.penup()
    moon.goto(x,y)
    moon.pendown()
    moon.circle(d)
    moon.end_fill()
    moon.hideturtle()    
drawMoon(0,0,100,'white')
drawStar(-100,-100,50,'brown')
drawStar(0,-100,50,'brown')
drawStar(100,-100,50,'brown')
t.exitonclick()
