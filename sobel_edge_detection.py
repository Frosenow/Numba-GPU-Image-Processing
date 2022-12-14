# Sobel-Filter Edge Detection Algorithm 
# Works only with grayscale images (2D arrays)
import numpy as np 
import matplotlib.pyplot as plt 
import skimage.data
from skimage.color import rgb2gray 
from PIL import Image 
from time import perf_counter_ns

def display_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show() 


# input image
# Image used for timing (3988x5982px)
img = rgb2gray(Image.open('4kMountains.jpg'))

# Image used for testing (512x512px)
# img = rgb2gray(skimage.data.coffee()) 
display_img(img, "Image used for edge detection")

vertical_sobel_filter = [[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]

horizontal_sobel_filter = [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]

n_rows, n_cols = img.shape

# VERTICAL EDGE DETECTION 
# vertical_edges_img = np.zeros_like(img)

# for row in range(3, n_rows - 2): # Ignoring the edges (TOP, BOTTOM)
#     for col in range(3, n_cols - 2): # Ignoring the edges (LEFT, RIGHT)
#         local_pixels = img[row-1:row+2, col-1:col+2] # Creating 3x3 box filter
#         transformed_pixels = vertical_sobel_filter * local_pixels 
#         vertical_score = (transformed_pixels.sum() + 4)/8 # Normalization to get values [0;1]
#         vertical_edges_img[row, col] = vertical_score

# display_img(vertical_edges_img, "After using vertical Sobel Filter")

# HORIZONTAL EDGE DETECION
# horizontal_edges_img = np.zeros_like(img)
# for row in range(3, n_rows - 2):
#     for col in range(3, n_cols - 2):
#         local_pixels = img[row-1:row+2, col-1:col+2] 
#         transformed_pixels = horizontal_sobel_filter * local_pixels 
#         horizontal_score = (transformed_pixels.sum() + 4)/8 
#         horizontal_edges_img[row, col] = horizontal_score

# display_img(horizontal_edges_img, "After using horizontal Sobel filter")

# APPLAYING BOTH FILTERS 
start_time = perf_counter_ns()

edges_img = np.zeros_like(img)

for row in range(3, n_rows - 2):
    for col in range(3, n_cols - 2):
        local_pixels = img[row-1:row+2, col-1:col+2]
        
        vertical_transformed_pixels = vertical_sobel_filter * local_pixels 
        vertical_score = (vertical_transformed_pixels.sum())/4
        
        horizontal_transformed_pixels = horizontal_sobel_filter * local_pixels 
        horizontal_score = (horizontal_transformed_pixels.sum())/4

        edge_score = (vertical_score ** 2 + horizontal_score ** 2)**.5
        edges_img[row, col] = edge_score

end_time = perf_counter_ns()
runtime = end_time - start_time
runtime *= 1e-9

display_img(edges_img, "After vertical and horizontal Sobel filter")
print(f"Estimated runtime: {runtime:.8f}")

