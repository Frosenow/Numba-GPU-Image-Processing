
import math
import numba 
import numpy as np 
import matplotlib.pyplot as plt
from numba import cuda
threads_16 = 16


@cuda.jit(device=True, inline=True)  # inlining can speed up execution
def amplitude(ix, iy):
    return (1 + math.sin(2 * math.pi * (ix - 64) / 256)) * (
        1 + math.sin(2 * math.pi * (iy - 64) / 256)
    )

# Example 2.5a: 2D Shared Array
@cuda.jit
def blobs_2d(array2d):
    ix, iy = cuda.grid(2)
    tix, tiy = cuda.threadIdx.x, cuda.threadIdx.y
    # print(ix, iy, tix, tiy)
    shared = cuda.shared.array((threads_16, threads_16), numba.float32)
    shared[tiy, tix] = amplitude(iy, ix)
    cuda.syncthreads()
    array2d[iy, ix] = shared[15 - tiy, 15 - tix]

# Example 2.5b: 2D Shared Array without synchronize
@cuda.jit
def blobs_2d_wrong(array2d):
    ix, iy = cuda.grid(2)
    tix, tiy = cuda.threadIdx.x, cuda.threadIdx.y

    shared = cuda.shared.array((threads_16, threads_16), numba.float32)
    shared[tiy, tix] = amplitude(iy, ix)

    # When we don't sync threads, we may have not written to shared
    # yet, or even have overwritten it by the time we write to array2d
    array2d[iy, ix] = shared[15 - tiy, 15 - tix]


N_img = 1024
blocks = (N_img // threads_16, N_img // threads_16)
threads = (threads_16, threads_16)

dev_image = cuda.device_array((N_img, N_img), dtype=np.float32)
dev_image_wrong = cuda.device_array((N_img, N_img), dtype=np.float32)

blobs_2d[blocks, threads](dev_image)
blobs_2d_wrong[blocks, threads](dev_image_wrong)

image = dev_image.copy_to_host()
image_wrong = dev_image_wrong.copy_to_host()


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.T, cmap="nipy_spectral")
ax1.set_title('Blocks distribution with sync.')
ax2.imshow(image_wrong.T, cmap="nipy_spectral")
ax2.set_title('Blocks distribution without sync.')
for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.show()