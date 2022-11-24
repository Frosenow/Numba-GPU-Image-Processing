import numpy as np 
import cupy as cp
from PIL import Image 
import time

# Wgrywanie obrazka i zmiana rozmiaru  
img = Image.open('parrot.jpg').resize((512, 512))

# Zamiana obrazka na tablice 
img_arr = np.asarray(img)

img_arr_CPU = img_arr 

# Wgrywanie danych do GPU 
img_arr_GPU = cp.asarray(img_arr)
