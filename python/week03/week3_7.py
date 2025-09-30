import tkinter as tk
from tkinter import messagebox

def on_button_click():
    messagebox.showinfo("알림","버튼이 클릭되었습니다.")

root = tk.Tk()
root.title("간단한 Tkinter 앱")
root.geometry("300x200")

btn = tk.Button(root,text="클릭하세요.",command=on_button_click)
btn.pack(pady=20)

root.mainloop()