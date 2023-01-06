import numpy as np 
import matplotlib.pyplot as plt
from numba import cuda
import skimage.data
from skimage.color import rgb2gray 
import math 
from matplotlib import image
from time import perf_counter_ns


def display_img(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show() 

@cuda.jit
def sobel_filter(img, result):
    x, y = cuda.grid(2)
    height, width = img.shape
    if x < height and y < width:
        # Compute the gradient magnitude using the Sobel operator
        dx = (img[x-1, y-1] + 2*img[x, y-1] + img[x+1, y-1]) - (img[x-1, y+1] + 2*img[x, y+1] + img[x+1, y+1])
        dy = (img[x-1, y-1] + 2*img[x-1, y] + img[x-1, y+1]) - (img[x+1, y-1] + 2*img[x+1, y] + img[x+1, y+1])
        # Multiply by 4 to rescale the gradient magnitudes 
        result[x, y] = 4 * math.sqrt(dx**2 + dy**2)   
    
# input image
# Image used for timing (3988x5982px)
img = rgb2gray(image.imread('4kMountains.jpg').astype(np.float32) / 255.)

# Image used for testing (512x512px)
# img = rgb2gray(skimage.data.coffee().astype(np.float32) / 255.)

display_img(img, "Image used for edge detection")

# Calculate the grid and block dimensions 
grid_dim = (int(np.ceil(img.shape[0] / 32)), int(np.ceil(img.shape[1] / 32)))
block_dim = (32, 32)

# Allocate space for the result on the device (GPU)
result_dev = cuda.device_array_like(img)

# Copy the input image to the device 
img_dev = cuda.to_device(img)

# Apply Sobel filter by calling the CUDA kernel 
sobel_filter[grid_dim, block_dim](img_dev, result_dev)
cuda.synchronize() #clear gpu from tasks

#measuring run time without compilation time
timing = np.empty(101)
for i in range(timing.size):
    start_time = perf_counter_ns()
    sobel_filter[grid_dim, block_dim](img_dev, result_dev)
    cuda.synchronize()
    end_time = perf_counter_ns()
    timing[i] = end_time-start_time
timing *= 1e-9

# Copy the result from device (GPU) back to the host 
result = result_dev.copy_to_host()

# Print output
display_img(result, 'Image after filtration')
print(f"Elapsed time: {timing.mean():.8f} +- {timing.std():.8f} s")