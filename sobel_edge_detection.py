# Sobel-Filter Edge Detection Algorithm 
# Works only with grayscale images (2D arrays)
import numpy as np 
import matplotlib.pyplot as plt 
import skimage.data
from skimage.color import rgb2gray 

def display_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show() 

img = rgb2gray(skimage.data.astronaut())
display_img(img, "Image used for edge detection")

vertical_sobel_filter = [[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]

horizontal_sobler_filter = [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]

n_cols, n_rows = img.shape

vertical_edges_img = np.zeros_like(img)

# VERTICAL EDGE DETECTION 
for row in range(3, n_rows - 2): # Ignoring the edges (TOP, BOTTOM)
    for col in range(3, n_cols - 2): # Ignoring the edges (LEFT, RIGHT)
        local_pixels = img[row-1:row+2, col-1:col+2] # Creating 3x3 box filter
        transformed_pixels = vertical_sobel_filter * local_pixels 
        vertical_score = (transformed_pixels.sum() + 4)/8 # Normalization to get values [0;1]
        vertical_edges_img[row, col] = vertical_score

display_img(vertical_edges_img, "After using vertical edge detection")

# HORIZONTAL EDGE DETECION
