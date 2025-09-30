#라디오 버튼 생성
import tkinter as tk

oroot=tk.Tk()
oroot.geometry("200x200")

radio_value = tk.IntVar()
radio_value.set(0)
lunch = {0:"A런치",1:"B런치",2:"C런치"}
orb1 = tk.Radiobutton(oroot,text=lunch[0],variable=radio_value,value=0)
orb2 = tk.Radiobutton(oroot,text=lunch[1],variable=radio_value,value=1)
orb3 = tk.Radiobutton(oroot,text=lunch[2],variable=radio_value,value=2)
orb1.pack()
orb2.pack()
orb3.pack()

def buy():
    value=radio_value.get()
    print(lunch[value])

obutton=tk.Button(oroot,text='주문',command=buy)
obutton.pack()

oroot.mainloop()