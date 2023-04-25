import sys, os
import numpy as np
from PIL import Image
import pickle


X, Y = [], []
labels = {"Cat":0,"Dog":1}
outDir = "features"
if not os.path.exists(outDir):
    os.mkdir(outDir)
outDirTotal = "features_minimal"
if not os.path.exists(outDirTotal):
    os.mkdir(outDirTotal)
fileTotal = os.path.join(outDirTotal, "Features")+".pickle"
np.random.seed(42)

def center_crop(image_path, size):
    img = Image.open(image_path)
    img = img.resize((size+1,size+1)) #101
    x_center = img.width/2 #50,5
    y_center = img.height/2 #50,5
    size = size/2 #50
    cr = img.crop((x_center-size, y_center-size, x_center+size, y_center+size))
                # 0,5; 0,5; 100,5; 100,5  
    return cr

#LoadImages
for root, directories, files in os.walk("AnimalFace"): # Thư mục gốc, thư mục con
    fileclass = os.path.join(os.path.abspath(outDir), os.path.split(root)[1])+'.pickle'
    Y_train = [] # label data list - Dữ liệu nhãn
    X_train = [] # image data list - Dữ liệu ảnh
    np.random.shuffle(files)
    for filename in files: # Duyệt từng ảnh trong từng thư mục con
        filepath = os.path.join(root, filename) # Nối đường dẫn đến thư mục con

        #CropImageToRGB
        crp_img = center_crop(filepath,100)
        #crp_img.show()
        r, g, b = crp_img.split() # partition the color scheme - Phân vùng bảng màu
        image = Image.merge("RGB", (g, g, g)).convert("L")
        getData = list(image.getdata())

        crp_arr = np.array(getData).reshape(-1)
        imgData = crp_arr/255

        X_train.append(imgData)
        Y_train.append(labels[os.path.split(root)[1]]) 

    if len(X_train) > 0:
        pickleF = open(fileclass, 'wb')
        pickle.dump(X_train, pickleF)
        pickle.dump(Y_train, pickleF)
        pickleF.close()
        print("")
        print("Dumping pickle to: "+fileclass)
        print("")

#LoadFeatures
for animal in os.listdir(outDir):
    filepath = os.path.join(outDir, animal)
    print("Reading features from: "+filepath)
    pickleF = open(filepath, 'rb')
    X.extend(pickle.load(pickleF))
    Y.extend(pickle.load(pickleF))
    pickleF.close()

#SavePickle
pickleF = open(fileTotal, 'wb')
pickle.dump(X, pickleF)
pickle.dump(Y, pickleF)
pickleF.close()
print("Features successfully dumped.")
print("Classes Loaded: "+str(len(set(Y))))

