import tkinter as tk

#윈도우 위젯 생성
pizza = tk.Tk()
pizza.geometry("400x300")
pizza.title("조각 피자 주문 프로그램")

pizza_label = tk.Label(pizza,text='피자')
pizza_label.pack(side='top')

#위젯 생성/배치
plist = { 0 : "치즈피자(3200원)" , 1 : "콤비네이션 피자(3500원)" , 2 : "불고기 피자(3600원)" }
clist = { 0 : 3200 , 1 : 3500 , 2 : 3600 }
check = {}

for i in range(len(plist)):
    check[i] = tk.IntVar() #tkinter에서 변수 생성 시 이 형식 사용!
    checkButton = tk.Checkbutton(pizza,text=plist[i],variable=check[i])
    checkButton.pack(anchor="nw")

#이벤트 바인딩 

def order():
    pizza_label = tk.Label(pizza,text='주문내역:')
    pizza_label.pack()
    sum = 0
    for i in check:
        if check[i].get() == 1:
            pizza_label = tk.Label(pizza,text=plist[i])
            pizza_label.pack()
            sum += clist[i]
    pizza_label = tk.Label(pizza,text="")
    pizza_label.pack()
    pizza_label = tk.Label(pizza,text=f"총 가격 : {sum} 원 ")
    pizza_label.pack()
    
pizza_button = tk.Button(pizza,text='주문',command=order)
pizza_button.pack()
pizza.mainloop()