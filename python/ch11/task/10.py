# 레이블에 이미지 추가

import tkinter as tk
oroot = tk.Tk()
oroot.geometry("200x100")
img1 = tk.PhotoImage(file = 'pizzasb2.png')
img_label=tk.Label(oroot,image=img1)
img_label.place(x=-20,y=-20)

obutton1 = tk.Button(oroot,text="push1")
obutton1.place(x=10,y=60)
obutton2 = tk.Button(oroot,text="push2")
obutton2.place(x=140,y=60)
obutton3 = tk.Button(oroot,text="push3")
obutton3.place(x=80,y=10)

oroot.mainloop()