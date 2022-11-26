from numba import cuda
import numpy as np 
import skimage.data 
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt

@cuda.jit 
def convolve(result, mask, image):
    i, j = cuda.grid(2)
    
    image_rows, image_cols = image.shape
    if (i >= image_rows) or (j >= image_cols): 
        return 

    delta_rows = mask.shape[0] // 2  # // divide with integral results
    delta_cols = mask.shape[1] // 2

    s = 0 
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k = i - k + delta_rows 
            j_l = j - l + delta_cols
            if (i_k >= 0) and (i_k <= image_rows) and (j_l >= 0) and (j_l < image_cols):
                s += mask[k, l] * image[i_k, j_l]
        
    result[i,j] = s 

full_image = rgb2gray(skimage.data.coffee()).astype(np.float32) / 255
plt.figure()
plt.imshow(full_image, cmap='gray')
plt.title("Full size image:")
image = full_image[150:350, 200:400].copy()

plt.figure()
plt.imshow(image, cmap='gray')
plt.title("part of the image we use:")
plt.show()

result = np.empty_like(image)

mask = np.random.rand(13, 13).astype(np.float32)
mask /= mask.sum()
print('Mask shape: ', mask.shape)
print('Mask first (3,3) elements: \n', mask[:3, :3])

blockdim = (32, 32)
print('Blocks dimensions: ', blockdim)

griddim = (image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] + 1)
print('Grid dimensions: ', griddim)

convolve[griddim, blockdim](result, mask, image)

plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Before convolution:') 
plt.figure() 
plt.imshow(result, cmap='gray')
plt.title('After convolution:')
plt.show()