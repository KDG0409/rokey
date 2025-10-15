## Entry
# import tkinter as tk
# t_window=tk.Tk()
# t_window.geometry("300x400")
# t_window.title("Entry")
# data=tk.StringVar()
# t_entry=tk.Entry(t_window,textvariable=data)
# t_entry.pack()
# t_window.mainloop()

##RadioButton 1개 선택(데이터 1개)
# import tkinter as tk
# t_window=tk.Tk()
# t_window.geometry("300x400")
# t_window.title("RadioButton")
# lunch = { 0: 'A',1:'B',2:'C' }
# data=tk.IntVar()
# data.set(0)
# t_rb1=tk.Radiobutton(t_window,text=lunch[0],variable=data,value=0)
# t_rb1.pack()
# t_rb2=tk.Radiobutton(t_window,text=lunch[1],variable=data,value=1)
# t_rb2.pack()
# t_rb3=tk.Radiobutton(t_window,text=lunch[2],variable=data,value=2)
# t_rb3.pack()
# def say():
#     n=data.get()
#     t_lab=tk.Label(t_window,text=lunch[n],)
#     t_lab.pack()
# t_b=tk.Button(t_window,text='확인',command=say)
# t_b.pack()
# t_window.mainloop()

##checkbutton 여러개 선택(데이터 여러개)
# import tkinter as tk
# t_window=tk.Tk()
# t_window.geometry("300x400")
# t_window.title("CheckButton")
# lunch = { 0: 'A',1:'B',2:'C' }
# data={}
# for i in range(len(lunch)):
#     data[i]=tk.BooleanVar()
#     data[i].set(0)
#     t_rb=tk.Checkbutton(t_window,text=lunch[i],variable=data[i])
#     t_rb.pack()
# def say():
#     for i in range(len(lunch)):
#         if data[i].get():
#             t_lab=tk.Label(t_window,text=lunch[i])
#             t_lab.pack()
# t_b=tk.Button(t_window,text='확인',command=say)
# t_b.pack()
# t_window.mainloop()
##OptionMenu 1개 선택
# import tkinter as tk
# t_window=tk.Tk()
# t_window.geometry("300x400")
# t_window.title("OptionMenu")
# lunch = ['A','B','C','D']

# data=tk.StringVar()
# data.set(lunch[0])
# t_om=tk.OptionMenu(t_window,data,*lunch)
# t_om.pack()

# def say():
#     dat=data.get()
#     t_lab=tk.Label(t_window,text=dat)
#     t_lab.pack()
# t_b=tk.Button(t_window,text='확인',command=say)
# t_b.pack()
# t_window.mainloop()

##이미지 추가
# import tkinter as tk
# t_window=tk.Tk()
# t_window.geometry("300x400")
# t_im=tk.PhotoImage('')
# t_lab=tk.Label(t_window,image=t_im)
# t_lab.pack()
# t_window.mainloop()