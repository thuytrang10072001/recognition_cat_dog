import numpy as np
import pickle
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog

labels = ["Cat","Dog"]

# load the model from disk
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

def center_crop(image_path, size):
    img = Image.open(image_path)
    img = img.resize((size+1,size+1))
    x_center = img.width/2
    y_center = img.height/2
    size = size/2
    cr = img.crop((x_center-size, y_center-size, x_center+size, y_center+size))
    return cr

if __name__ == '__main__':

    root = Tk()
    root.geometry("350x350+300+150")
    root.title("Sklearn")
    root.resizable(width=True, height=True)
 
    def openfn():
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img():
        x = openfn()

        image_pad = []
        image = Image.open(x) # open image in binary
        im = center_crop(x,100) # cropping image
        r, g, b = im.split() # partition the color scheme - Phân vùng bảng màu
        image_ = Image.merge("RGB", (g, g, g)).convert("L")
        getData = list(image_.getdata())
        X = np.array(getData).reshape(1,-1) # flattening the image to pass in model for prediction
        image_pad = X/255

        pred = model.predict(image_pad)
        print(pred)
        print(labels)
        print("Predict %s" % (labels[pred[0]]))

        lbl1 = Label(root, text = "Predict: %s" % (labels[pred[0]]), font = ("Palatino Linotype",11,"bold"))
        lbl1.place(x = 40, y = 310)

        img = Image.open(x)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        panel.place(x = 50, y = 40)

    btn = Button(root, text='Open Image', command=open_img, font = ("Palatino Linotype",11,"bold") ).pack()
    root.mainloop()