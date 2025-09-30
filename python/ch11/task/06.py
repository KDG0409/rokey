# import tkinter as tk #grid 메소드 사용,행렬기준

# oroot = tk.Tk()
# oroot.geometry("200x100")

# obutton1 = tk.Button(oroot,text="push1")
# obutton1.grid(row=1,column=0)
# obutton2 = tk.Button(oroot,text="push2")
# obutton2.grid(row=1,column=1)
# obutton3 = tk.Button(oroot,text="push3")
# obutton3.grid(row=0,column=3)

# oroot.mainloop()

# import tkinter as tk #pack 메소드 사용, 좌우상하 기준

# oroot = tk.Tk()
# oroot.geometry("200x100")

# obutton1 = tk.Button(oroot,text="push1")
# obutton1.pack(side='left')
# obutton2 = tk.Button(oroot,text="push2")
# obutton2.pack(side='right')
# obutton3 = tk.Button(oroot,text="push3")
# obutton3.pack(side='top')

# oroot.mainloop()

import tkinter as tk #place 메소드 사용, 좌표 기준

oroot = tk.Tk()
oroot.geometry("200x100")

obutton1 = tk.Button(oroot,text="push1")
obutton1.place(x=10,y=60)
obutton2 = tk.Button(oroot,text="push2")
obutton2.place(x=140,y=60)
obutton3 = tk.Button(oroot,text="push3")
obutton3.place(x=80,y=10)

oroot.mainloop()