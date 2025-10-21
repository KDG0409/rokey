#체크버튼 생성
import tkinter as tk

oroot=tk.Tk()
radio_value = tk.IntVar()
lunch = {0:"A런치",1:"B런치",2:"C런치"}
check_value={}

for i in range(len(lunch)):
    check_value[i] = tk.BooleanVar()
    ocheckbutton = tk.Checkbutton(oroot,variable = check_value[i],text=lunch[i])
    ocheckbutton.pack(anchor="w")

def buy():
    for i in check_value:
        if check_value[i].get()==True:
            print(lunch[i])

tk.Button(oroot,text='주문',command=buy).pack()
oroot.mainloop()