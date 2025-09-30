# import tkinter as tk

from tkinter import* #(코드 간결화)

otk = Tk() #윈도우 위젯(최상위) 객체 생성
obtn = Button(otk,text='click') #버튼 위젯 객체 생성 (버튼클래스)
obtn.pack() #버튼 위젯 객체 배치
otk.mainloop() #윈도우 위젯 무한 반복(활성화 유지)