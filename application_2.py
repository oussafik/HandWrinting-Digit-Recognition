
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
import cv2 as cv

my_w = tk.Tk()
model=load_model('mnist.h5')
my_w.geometry("400x300")  # Size of the window
my_w.title('Digit Recognition')
my_font1=('times', 18, 'bold')
my_font2=('normal', 15, 'bold')
l1 = tk.Label(my_w,text='Importer une Image',width=30,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Importer', width=20,command = lambda:upload_file())
b1.grid(row=2,column=1)
l2 = tk.Label(my_w,text='',width=28,font=my_font2)
l2.place(x=30,y=265)
def resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv.resize(img, size, cv.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv.INTER_AREA if dif > (size[0]+size[1])//2 else cv.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv.resize(mask, size, interpolation)


"""explication: la fonction prend une entrée de n'importe quelle taille et crée une image vierge 
de forme carrée de la hauteur ou de la largeur de l'image de taille, selon ce qui est le plus grand. 
il place ensuite l'image d'origine au centre de l'image vierge. puis il redimensionne cette 
image carrée à la taille souhaitée afin que la forme du contenu de l'image d'origine soit préservée."""


def upload_file():
    global img
    f_types = [('Png Files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    img_resized=img.resize((300,200)) # new width & height
    img=ImageTk.PhotoImage(img_resized)
    b2 =tk.Button(my_w,image=img) # using Button
    b2.grid(row=3,column=1)
    img2 = cv.imread(filename)[:, :, 0]
    img2 = resize_image(img2)
    img2 = np.invert(np.array([img2]))
    prediction = model.predict(img2)
    l2["text"] = 'Le nombre est : '+ str(np.argmax(prediction))


my_w.mainloop()  # Keep the window open

