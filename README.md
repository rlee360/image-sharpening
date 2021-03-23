# ECE453: Advanced Computer Architecture Project 1: Image Deblur

The goal of this project was to become familiar with CUDA C toolchain and how to use it to perform highly parallelizable tasks, such as image processing. This program uses a Laplacian Sharpening Kernel to sharpen images, and computes evaluation metric

# Dependencies

Requires:
 
 + `stb_image.h` and `stb_image_write.h` (included in the repository)
 + `libpopt` and `libpopt-dev` (on Debian-based system, this can be installed using `sudo apt install libpopt0 libpopt-dev`)
 

# Compiling Instructions

The provided Makefile assumes a Jetson platform, and will build everything.

```bash
$ make
```

Alternatively, compilation can be done on any CUDA-capable machine using nvcc:

```bash
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} # set path variables, if not already
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} # set path variables, if not already
$ nvcc deblur.cu -o deblur -lm -lpopt
```

# Usage Instructions

The program requires a space separated file that denotes the convolution kernel to use to perform sharpening. The first line is a single integer that denotes the size of the mask in both dimensions, while subsequent lines denote the actual mask. A 7x7 kernel is included in this repository and was used for the testing of the program.

The program defaults to using the provided blurred image of `nasa_earth_blurry.png` and reference image of `nasa1R13107_1000x1000.jpg`, as well as running the sharpening on both the CPU and GPU. These, however, like the kernel, can be configured using command line options:

```
Usage: deblur [OPTION...]
  -k, --kernel=kernel_file              Space separated file designating the
                                        deconvolution kernel. First line is a
                                        single INT that describes the size of
                                        the kernel. Subsequent lines describe
                                        the actual kernel.
  -i, --input=blurred_image_file        Image to deblur
  -r, --reference=sharp_image_file      Non-blurred, sharp image. Used to
                                        assess sharpness. If not provided,
                                        error will not be computed
  -o, --output_prefix=prefix_string     Output image prefix. Will be followed
                                        by _cpu.png/_gpu.png. Default: "output"
  -d, --device=gpu|cpu|both             Specify to run on gpu, cpu, or both.
                                        Default: both

Help options:
  -?, --help                            Show this help message
      --usage                           Display brief usage message

```

Sharpening the provided `images/nasa_earth_blurry.png` on both gpu and cpu:

```
$ ./deblur --input images/nasa_earth_blurry.png --reference images/nasa1R3107_1000x1000.jpg --output_prefix output/nasa_output
```

As an example of sharpening the included `images/tree-blur.jpg` on the GPU only:

```
$ ./deblur -i images/tree-blur.jpg -r images/tree-clear.jpg -o output/tree_output -d gpu
```