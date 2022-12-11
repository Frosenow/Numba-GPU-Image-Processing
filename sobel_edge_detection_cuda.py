import numba 
import numpy as np 
import matplotlib.pyplot as plt
from numba import cuda
import matplotlib.pyplot as plt 
import skimage.data
from skimage.color import rgb2gray 


def display_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show() 

img = rgb2gray(skimage.data.coffee())

Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

[rows, columns] = np.shape(img)
sobel_filtered_image = np.zeros(shape=(rows, columns))

for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  # x direction
        gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  # y direction
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"



