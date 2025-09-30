import tkinter as tk

pizza = tk.Tk()
pizza.geometry("400x300")
pizza.title("조각 피자 주문 프로그램")
pizza_label = tk.Label(pizza,text="피자")
pizza_label.pack(side="top")

plist = { 0 : "치즈 피자 (3200원)" , 1 : "콤비네이션 피자 (3500원)" , 2 : "불고기 피자 (3600원)" }
clist = { 0 : 3200 , 1 : 3500 , 2 : 3600 }
check= {}

for i in range(len(plist)):
    check[i] = tk.IntVar()
    checkButton = tk.Checkbutton(pizza,text=plist[i],variable=check[i])
    checkButton.pack(anchor = "nw")

def order():
    mainLabel = tk.Label(pizza,text="주문내역:")
    mainLabel.pack()
    sum = 0
    for i in check:
        if check[i].get() == True :
            orderLabel = tk.Label(pizza,text=plist[i])
            orderLabel.pack()
            sum += clist[i]
    enterLabel = tk.Label(pizza,text="")
    enterLabel.pack()
    costLabel = tk.Label(pizza,text=f"총 가격:{sum}")
    costLabel.pack()
    
orderButton = tk.Button(pizza,text="주문",command = order)
orderButton.pack()

pizza.mainloop()