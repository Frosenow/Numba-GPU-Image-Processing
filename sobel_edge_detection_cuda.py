import numpy as np 
import matplotlib.pyplot as plt
from numba import cuda
import skimage.data
from skimage.color import rgb2gray 
import math 


def display_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show() 

@cuda.jit
def sobel_filter(img):
    y, x = cuda.grid(2)
    if x >= img.shape[1] or y >= img.shape[0]:
        return

    # compute Sobel filter
    sobel_x = (img[y-1, x-1] + 2*img[y, x-1] + img[y+1, x-1]) - (img[y-1, x+1] + 2*img[y, x+1] + img[y+1, x+1])
    sobel_y = (img[y-1, x-1] + 2*img[y-1, x] + img[y-1, x+1]) - (img[y+1, x-1] + 2*img[y+1, x] + img[y+1, x+1])
    img[y, x] = math.sqrt(sobel_x**2 + sobel_y**2)
    
# input image
img = rgb2gray(skimage.data.coffee().astype(np.float32) / 255.)
height, width = img.shape

threads = 256
threads_per_block = (threads, threads); 
blocks_per_grid = (height // threads, width // threads)

# apply Sobel filter
filtered_img = cuda.to_device(img)
sobel_filter[(64, 64), (16, 16)](filtered_img)
filtered_img.copy_to_host()

# print output
display_img(filtered_img, 'title')
