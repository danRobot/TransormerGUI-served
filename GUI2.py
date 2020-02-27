from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
from googletrans import Translator
import asyncio
from client.client import hello

class App:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, width=1280, height=768)
        
        self.canvas.pack()

        #load = Image.open("fondo.png")
        #render = ImageTk.PhotoImage(load)
        #img = Label(self.canvas, image=render)
        #img.image = render
        #img.place(x=0, y=0)
        self.canvas.configure(bg="#1f2021")
    
        explicacion = Label(self.canvas, text="Ingresa frase inicial:", background="#1f2021", foreground = "white", font="Helvetica 12 bold")
        explicacion.place(x=48, y=20)

        txt_generado = Text(self.canvas)
        txt_generado.config(width=78,height=38, background="white", foreground = "#171717")
        txt_generado.place(x=608, y=100)

        txt_ingresado = Entry(self.canvas,width=180)
        txt_ingresado.place(x=48, y=50)

        def clicked():
            txt_generado.delete('1.0', END)
            translator = Translator()
            txt_ingresado_ingles=translator.translate(txt_ingresado.get(),dest='en').text
            txt_generado_ingles = generar_texto(txt_ingresado_ingles)
            temp_txt=translator.translate(txt_generado_ingles, dest='es').text
            txt_generado.insert(INSERT, temp_txt)

        def generar_texto(entrada_texto):
            salida_texto=asyncio.get_event_loop().run_until_complete(hello(entrada_texto    ))
            return salida_texto

        btn = Button(self.canvas, text="Generar", command=clicked)
        btn.place(x=1160, y=48)

        self.sequence = [ImageTk.PhotoImage(img)
                            for img in ImageSequence.Iterator(
                                    Image.open(
                                    r'sdf.gif'))]

        self.image = self.canvas.create_image(320,408, image=self.sequence[0])
        
        self.animate(1)


    def animate(self, counter):
        self.canvas.itemconfig(self.image, image=self.sequence[counter])
        self.parent.after(20, lambda: self.animate((counter+1) % len(self.sequence)))




root = tk.Tk()
root.title("Proyecto Shakespeare")

app = App(root)
root.mainloop()
