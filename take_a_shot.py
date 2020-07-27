from tkinter import *
from tkinter import messagebox
from tkinter import *
import cv2
from PIL import Image, ImageTk
import cv2
import time
import webbrowser
# 註冊網站
url = 'https://www.google.com.tw/?hl=zh-TW'
root = Tk()
root.title("Take a shot")

def take_shot():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    counter = 0
    while (counter < 3):
        # get a frame
        ret, frame = cap.read()
        cv2.waitKey(1000)
        img = cv2.flip(frame, 1)
        # show a frame
        cv2.imshow("Take a shot", img)
        # 拍照後存檔
        directory = "E:/user/Desktop/" + str(var.get()) + ".jpg"
        cv2.imwrite(directory, img)
        counter = counter + 1
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)
    register()

def OpenUrl():
    webbrowser.open_new(url)

def pop_up():
    messagebox.showinfo("", "Hello, "+var.get()+", 三秒後拍照")
    time.sleep(1)
    take_shot()

def register():
    win2 = Tk()
    win2.title("Register")
    info = Label(win2, text="前往註冊", font=18)
    info.pack(side=TOP)
    button_register = Button(win2, text="YES",width=15, command=OpenUrl)
    button_register.pack(side=LEFT)
    button_register_no = Button(win2, text="NO",width=15, command=win2.destroy)
    button_register_no.pack(side=RIGHT)
    root.destroy()


lbl1 = Label(root, text="Enter your name:",font=('Courier',18))
lbl1.pack(side=LEFT)
var = StringVar()
ent1 = Entry(root, bd=5,font=16, textvariable=var).pack(side=LEFT)
button_pop = Button(root, text="Done", font=('Courier',18), command=pop_up)
button_pop.pack(side=RIGHT)
root.mainloop()






