#1. b
#2. c
#3. b
#4. d
#5/6/7/8/9/10.
import tkinter as tk
t_window = tk.Tk()
t_window.title("조각피자 주문 프로그램")
t_window.geometry("400x300")
t_lab=tk.Label(t_window,text='피자').pack()
menu={0:'치즈피자 (3200)원',1:'콤비네이션 피자 (3500원)',2:'불고기 피자 (3600원)'}
cost={0:3200,1:3500,2:3600}
data={}
for i in range(len(menu)):
    data[i]=tk.BooleanVar()
    data[i].set(0)
    t_ch=tk.Checkbutton(t_window,text=menu[i],variable=data[i])
    t_ch.pack()
def pay():
    t_label=tk.Label(t_window,text='주문내역:')
    t_label.pack()
    total=0
    for i in range(len(menu)):
        if data[i].get():
            t_lab=tk.Label(t_window,text=menu[i])
            t_lab.pack()
            total+=cost[i]
    t_label=tk.Label(t_window,text='')
    t_label.pack()
    t_label=tk.Label(t_window,text=total)
    t_label.pack()
t_bt=tk.Button(t_window,text='주문',command=pay).pack()
t_window.mainloop()
