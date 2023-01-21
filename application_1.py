from PIL import ImageTk,Image
import PIL.Image
import io
from tkinter import *
import tkinter as tk
import numpy as np
from keras.models import load_model
from PIL import EpsImagePlugin

windo = Tk()
windo.configure(background='white')
windo.title("Reconnaissance des chiffres")
model = load_model('mnist.h5')

windo.geometry('1120x820')
windo.iconbitmap('./meta/wr.ico')
windo.resizable(0,0)

EpsImagePlugin.gs_windows_binary = r'C:\Program Files (x86)\gs\gs10.00.0\bin\gswin32c'


def destroy_widget(widget):
    widget.destroy()


def pred_digit():
    global no,no1
    ps = canvas.postscript(colormode='color')
    # use PIL to convert to PNG
    im1 = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
    img = im1.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    pred = np.argmax(res)
    acc = max(res)
    no = tk.Label(windo, text='Le chiffre est: '+str(pred), width=34, height=1,
                  fg="white", bg="midnightblue",
                  font=('times', 16, ' bold '))
    no.place(x=460, y=380)

    no1 = tk.Label(windo, text='La précision est: '+str(acc), width=34, height=1,
                   fg="black", bg="yellow",
                   font=('times', 16, ' bold '))
    no1.place(x=460, y=415)

def draw_digit(event):
#    canvas.configure(background="black")
    x = event.x
    y = event.y
    r=10
    canvas.create_oval(x-r, y-r, x + r, y + r, fill='black')
    panel5.configure(state=NORMAL)

def clear_digit():
    panel5.configure(state=DISABLED)
    canvas.delete("all")
    try:
        no.destroy()
        no1.destroy()
    except:
        pass

im = PIL.Image.open('./meta/image-removebg-preview.png')
im =im.resize((311,283), PIL.Image.ANTIALIAS)
wp_img = ImageTk.PhotoImage(im)
panel4 = Label(windo, image=wp_img,bg = 'white')
panel4.pack()
panel4.place(x=20, y=20)

panel5 = Button(windo,text = 'Reconnaître',state=DISABLED,command = pred_digit,width = 15,borderwidth=0,bg = 'midnightblue',fg = 'white',font = ('times',18,'bold'))
panel5.place(x=60, y=305)

panel6 = Button(windo,text = 'Supprimer',width = 15,borderwidth=0,command = clear_digit,bg ='yellow',fg = 'black',font = ('times',18,'bold'))
panel6.place(x=60, y=355)

canvas = tk.Canvas(windo, width=405, height=280,highlightthickness=1, highlightbackground="midnightblue", cursor="pencil")
canvas.grid(row=0, column=0, pady=2, sticky=W,)
canvas.place(x=460,y=90)
canvas.bind("<B1-Motion>", draw_digit)

lab = tk.Label(windo, text="Ecrivez un chiffre", width=18, height=1, fg="white",bg="midnightblue",font=('times', 16, ' bold '))
lab.place(x=554, y=60)

windo.mainloop()