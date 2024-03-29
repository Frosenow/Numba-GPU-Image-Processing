<h1>Project Description and Technology Used</h1>
<p align="justify">In this project, research was conducted on the performance of a program in parallel mode, in comparison to its performance in single-thread mode. The research was conducted on an edge detection algorithm. The project was implemented using the CUDA library. NVIDIA CUDA (Compute Unified Device Architecture) is a platform for parallel computing and an application programming interface (API) that allows software to use specific types of graphics processing units (GPUs) for general-purpose processing, an approach known as general-purpose computing on GPUs. CUDA is a software layer that provides direct access to the virtual instruction set of the GPU and parallel computing elements, in order to perform computational kernels.</p>

<h2>Problem Description - Edge Detection Algorithm Using Sobel Operator</h2> 
<p align="justify">The basic and simplest solution to the problem of creating an edge detection algorithm that uses the Sobel operator is the "naive" approach. However, this approach for images with high resolutions, such as 3840 x 2160 pixels (4K) and more, requires many calculations that take a lot of time. So, what does the "naive" implementation of the algorithm involve?

The naive implementation is based on the assumption that one working element generates one filtered output pixel. To filter each pixel, each working element must load a 3x3 matrix of pixels in such a way that the middle pixel corresponds to the main element, plus another 8 neighboring pixels. If we assume that our input data is a 2D grayscale image with dimensions W x H, where W is the width and H is the height (in pixels), in this case, the main function, i.e., the kernel, must run W * H working elements, which for a 4K image gives 3840 * 2160 = 8294400 working elements. It is worth noting that this is a situation where the used image is a grayscale image, not an RGB image that has 3 layers.

The implemented algorithm in a single-threaded version (executed on a CPU) is an inefficient algorithm, especially for large images as mentioned earlier. In summary, the biggest cost incurred is the convolution of the image with Sobel filters. Each pixel in the output edge map requires a 3x3 matrix consisting of the input image and the appropriate multiplication by the Sobel filter. The data division in the case of edge detection is as follows: the input data can be divided into smaller blocks, each of which is processed by a separate thread. In this case, the block size should be selected to match the size of the Sobel operator, which will allow for edge detection operations to be performed on the entire image area without creating "dead" areas.</p>

<h3><a href="https://github.com/Frosenow/Numba-GPU-Image-Processing/blob/main/sobel_edge_detection.py">Single-threaded algorithm project</a></h3>
<p align="justify">The single-threaded implementation of the edge detection algorithm is based on using the Sobel operator, which is a popular method for detecting edges in images. The algorithm starts with loading the original image and then converting it to grayscale. Two Sobel filters are then initialized: one for detecting vertical edges and one for detecting horizontal edges. These filters are 3x3 matrices that are convolved with the input image to produce two separate edge maps, one for vertical edges and one for horizontal edges. The vertical edge map is created by convolving the vertical Sobel filter with the input image, and the horizontal edge map is created by convolving the horizontal Sobel filter with the input image. Convolution is performed by iterating through the rows and columns of the image, creating a 3x3 window around each pixel, and multiplying the values in the window by the corresponding values in the Sobel filter. The resulting sum is stored in the appropriate location in the edge map. The algorithm then combines the two edge maps to obtain the final edge map by calculating the gradient magnitude, which is the square root of the sum of the squares of the gradient in the x direction and the gradient in the y direction. Finally, the image is averaged to prevent it from appearing darker than the original image.</p>
<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227617175-932724d0-6556-4c24-8e72-5d1ec753d346.png" alt="single-thread-diagram"/>
</div>

<h3><a href="https://github.com/Frosenow/Numba-GPU-Image-Processing/blob/main/sobel_edge_detection_cuda.py">Multi-threaded algorithm project</a></h3>
<p align="justify">A multi-threaded implementation of the edge detection algorithm based on the Sobel operator has been created using the CUDA library. The algorithm starts by defining a CUDA kernel function that is executed on the graphics processing unit (GPU). The kernel function takes two arguments: the input image and an empty array to store the result, as the kernel function cannot directly return a result. The kernel function is designed to operate on a 2D thread grid, where each thread operates on a single pixel of the input image. The kernel function uses the Sobel operator to calculate the gradient magnitude of each pixel, which is used as the edge strength. The Sobel operator is a 3x3 convolution filter used to calculate the gradient magnitude at each pixel. The gradient is multiplied by a constant value of 4, and normalization is optional depending on the desired output range. Multiplying by a value of 4 is sufficient to keep the output values in the range of [0,4]. The Sobel operator uses two kernels, one for detecting horizontal edges and one for detecting vertical edges. The horizontal kernel is [-1, 0, 1], [-2, 0, 2], [-1, 0, 1], and the vertical kernel is [-1, -2, -1], [0, 0, 0], [1, 2, 1]. The kernel is convolved with the input image by multiplying the corresponding elements and summing the values, which are then used to calculate the gradient magnitude. After defining the kernel function, the algorithm reads the input image, which should be in grayscale. Then it initializes the thread grid and block dimensions for the CUDA kernel. The grid dimensions are calculated by dividing the image size by the block size and correspond to the number of blocks that can fit on the grid, and the block dimensions are set to a fixed value (32, 32) determined by previous algorithm tests with other values and image sizes. The block size corresponds to the number of threads in each block. The algorithm then allocates memory for the result on the GPU, copies the input image to the GPU, and applies the Sobel filter by invoking the CUDA kernel. The CUDA kernel is executed on the GPU using the calculated grid and block dimensions. The kernel function is executed multiple times to measure the execution time without the compilation time. After the kernel execution is completed, the result is copied back to the host (CPU) and displayed. Using CUDA kernels in this way allows the algorithm to take advantage of the parallel processing capabilities of the GPU. Since the Sobel edge detection algorithm is computationally intensive, running it on the GPU can significantly improve its performance, especially for large images. The CUDA kernel can be executed in parallel on multiple threads, which can significantly reduce its execution time. In addition, using the Numba library and the @cuda.jit function decorator allows for compiling and executing functions on the GPU without using explicit CUDA code.</p>
<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227629635-05ed165d-b734-45bc-832e-62b7bbaae025.png" alt="nulti-thread-diagram"/>
</div>

<h2>Time measurement</h2> 
<p align="justify">A program was written to measure the execution time of the algorithm using CUDA technology and single-threaded execution of the algorithm on the processor alone. Measurements were taken without considering the time required to initialize the data, only the time taken to execute the algorithm. A grayscale image with a resolution of 3988x5982 pixels was used as the object for the measurements.
First, the execution time of the single-threaded version of the algorithm was measured. The time taken for one iteration on Device 1 was 262.451s, and on Device 2 it was 244.508s. Based on the measurements, it can be concluded that Device 2's processor is 107% more efficient than Device 1's processor.
Next, a measurement was taken using CUDA. The measurement was carried out for 100 iterations, and the average runtime and standard deviation were calculated. The average time taken for one iteration on Device 1 was 0.01611751s ± 0.00143468s, and on Device 2 it was 0.01665402s ± 0.00087353s. It can be observed that this time, Device 1 has greater computing power than Device 2, by 96%.
Multi-threaded execution was significantly faster than single-threaded execution, as expected. For Device 1, the runtime was reduced by a factor of 16283, and for Device 2, the runtime was reduced by a factor of 14680.
The data was presented in Graph 1. To better illustrate the results for CUDA, the runtime was magnified by a factor of 1e3.</p>
<ul>
  <li>Device 1:  CPU: Intel Core i5-11400h, GPU: NVIDIA GeForce RTX3050 </li>
  <li>Device 2:  CPU: AMD Ryzen 9 5900x, GPU: NVIDIA GeForce GTX970</li>
</ul>

<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227633813-efe8397e-c0ec-4d54-af78-db22946e845f.png" alt="comparison-diagram"/>
  <span>Graph 1. Single-threaded vs Multi-threaded comparison</span>
</div>

<h2>Study of the effect of block size on the execution time of a multithreaded algorithm</h2>
<p align="justify">The initialization of the grid and block dimensions for the CUDA kernel is calculated by dividing the image size by the block size, and the block dimensions are set arbitrarily before running the algorithm. The results of the study below are aimed at examining how the block size affects the speed of algorithm execution. The block size [32 x 32] is the maximum value that could be achieved while maintaining the correct operation of the previously implemented algorithm.

The number of blocks that can be used depends on the dimensions of the grid, and each CUDA block consists of a group of threads. Then, the kernel is executed as a grid of blocks containing threads. Each CUDA block is executed by one streaming multiprocessor (SM) and cannot be transferred to other SMs in the GPU, while one SM can run several concurrent CUDA blocks depending on the required resources.
---
<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227637175-1cb20798-748b-4c02-8694-d935c4cd9355.png" alt="blocks-size-diagram"/>
  <span>Graph 2. Demonstrate the effect of block size on calculation execution time</span>
</div> 

---

Block size    | Time of execution [s]
------------- | -------------------
1 x 1         | 0.34393714 ± 0.00015686
2 x 2         | 0.08736010 ± 0.00090976
4 x 4         | 0.02264809 ± 0.00089678
8 x 8         | 0.01108333 ± 0.00048379
16 x 16       | 0.01115149 ± 0.00006998
32 x 32       | 0.01615878 ± 0.00007989

---

As shown in the graph above, the computation execution time decreases rapidly as the block size increases. With the increase in block size, the CUDA algorithm can utilize increased parallelism and potentially work faster. However, when the block size becomes very large, the overall costs associated with managing the blocks and communicating between threads within the block may become significant in relation to the computations performed, leading to a decrease in performance. Therefore, there is often an optimal block size that balances the benefits of increased parallelism with the overhead and shared memory conflicts. The optimal block size depends on the specific problem being solved and the hardware on which the algorithm is running.</p>

<h2>The result of the algorithm</h2>


<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227638295-d4543796-26f8-4765-9984-b4c7180a0aa8.jpg" alt="before-filtration" width=398 heigth=598/>
  <br>
  <span>Image 1. Image before edge detection algorithm</span>
</div> 
<div align="center">
  <img src="https://user-images.githubusercontent.com/75395761/227638142-0dbd5447-0865-4d19-9cb8-ce64dd300a92.png" alt="filtered-image"/> 
  <br>
  <span>Image 2. Image after edge detection algorithm</span>
</div> 

## Examples:
Take a look at these couple examples that I have in my own portfolio:

**Interactive Linear Regression using Tensorflow.js:** https://github.com/Frosenow/Learning-Tensorflowjs

**Image Processing in Python using CUDA with Numba:** https://github.com/Frosenow/Numba-GPU-Image-Processing

**Red Planet Scout - Images from Mars:** https://github.com/Frosenow/RedPlanetScout---Mars-Photos-Gallery





