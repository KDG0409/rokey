#1. b
#2. c
#3. b
#4. d
##5/6/7/8/9/10. 
import tkinter as tk
t_window = tk.Tk()
t_window.geometry('400x300')
t_window.title('조각 피자 주문 프로그램')

t_label = tk.Label(t_window,text='피자')
t_label.pack()

menu = {0:'치즈 피자 (3200원)',1:'콤비네이션 피자 (3500원)',2:'불고기 피자 (3600원)'}
cost = [3200,3500,3600]
data={}

for i in range(len(menu)):
    data[i] = tk. BooleanVar()
    t_check=tk.Checkbutton(t_window,text=menu[i],variable=data[i])
    t_check.pack()

def buy():
    t_label = tk.Label(t_window,text='주문내역:')
    t_label.pack()
    total = 0
    for i in range(len(data)):
        if data[i].get() == 1 :
            t_label = tk.Label(t_window,text=f'-{menu[i]}')
            t_label.pack()
            total += cost[i]
    t_label = tk.Label(t_window,text='')
    t_label.pack()
    t_label = tk.Label(t_window,text=f' 총 가격 : {total}원 ')
    t_label.pack()

t_button = tk.Button(t_window,text='주문',command=buy)
t_button.pack()

t_window.mainloop()