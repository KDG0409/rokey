import tkinter as tk

oroot = tk.Tk()

olabel1 = tk.Label(oroot,text='적',bg='red',width=20)
olabel2 = tk.Label(oroot,text='녹',bg='green',width=20)
olabel3 = tk.Label(oroot,text='파',bg='blue',width=20)
olabel1.pack()
olabel2.pack()
olabel3.pack()

oroot.mainloop()
