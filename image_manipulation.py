from numba import cuda
import numpy as np 
import skimage.data 
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt
import math


@cuda.jit 
def convolve(image, result):
    i, j = cuda.grid(2) 

    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)
    image_rows, image_cols = image.shape
    if (i >= image_rows) or (j >= image_cols):
        return 
    
    for i0 in range(j, image_rows, threads_per_grid_y):
        for i1 in range(i, image_cols, threads_per_grid_x):
            result[i0, i1] =  10 * math.log2(1 + image[i0, i1])
    


full_image = rgb2gray(skimage.data.astronaut())
plt.figure()
plt.imshow(full_image, cmap='gray')
plt.title("Full size image:")
image = full_image[150:350, 200:400].copy()

plt.figure()
plt.imshow(image, cmap='gray')
plt.title("part of the image we use:")
plt.show()

result = np.empty_like(image)

imgBlockDim = (32, 32)
print('Image is divided for blocks size: ', imgBlockDim)
gridDim = (image.shape[0] // imgBlockDim[0] + 1, image.shape[1] // imgBlockDim[1] + 1)

d_image = cuda.to_device(image) 
d_result = cuda.to_device(result)
convolve[gridDim, imgBlockDim](d_image, d_result)

result = d_result.copy_to_host()


plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Before convolution:') 
plt.figure() 
plt.imshow(result, cmap='gray')
plt.title('After convolution:')
plt.show()
print(result)
