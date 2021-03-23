/**
  The 6th parameter of the stbi_write_png function from the above code is the
image stride, which is the size in bytes of a row of the image. The last
parameter of the stbi_write_jpg function is a quality parameter that goes from 1
to 100.
**/
#include <inttypes.h>
#include <math.h>
#include <popt.h> // requires compile flag -lpopt // Install on Debian systems: `sudo apt install libpopt0 libpopt-dev`
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

#define MAXFILENAMELEN 100

// struct to pass image parameters around easily.
typedef struct sharpen_metadata {
    int width;           // width of image in pixels
    int height;          // height of image in pixels
    int channels;        // number of image channels
    int kernel_size;     // number of elements in kernel in each dim
    int padding;         // width of padding on each edge
    int padded_width;    // width of image after adding padding
    int padded_height;   // width of image after adding padding
    int num_pix;         // number of total pixels in image
    int num_padded_pix;  // number of padded pixels in image
    int num_float_bytes; // size of padded image. sizeof(float) * num_padded_pix
} sharpen_metadata;

/************************** Error Handling Functions **************************/

void error_print(int ret_val, int sets_perror, int exit_code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    if (ret_val < 0) {
        vfprintf(stderr, format, args);
        if (sets_perror) perror("");
        va_end(args);
        if (exit_code) exit(exit_code);
    }
    va_end(args);
}

void* malloc_err(size_t _num_bytes, const char* file, int line) {
    void* _mem = malloc(_num_bytes);
    if (_num_bytes != 0) {
        error_print(_mem == NULL ? -1 : 0, 1, 7, "Error calling malloc in file %s on line %d\n: ", file, line);
    }
    memset(_mem, 0, _num_bytes);
    return _mem;
}

#define MALLOC_ERR(_size) (malloc_err(_size, __FILE__, __LINE__))

void cuda_check_error(cudaError_t _err, const char* _file, int _line) {
    error_print(_err != cudaSuccess ? -1 : 0, 0, _err, "Error in Cuda call in file %s on line %d: %s\n", _file, _line,
                cudaGetErrorString(_err));
}

#define CUDA_CHECK_ERROR(err) (cuda_check_error(err, __FILE__, __LINE__))

/************************* Image Processing Functions *************************/

// some automated math to simplify indexing of a pixel in a channels last without the overhead of many function calls
#define IMGIDX(idx_r, idx_c, idx_d, w, ch, p, i, j) ((idx_r + i + p) * (w) * (ch) + (idx_c + j + p) * (ch) + (idx_d))

// One thread should exist for each valid pixel in the padded image. This is made into a row, column, and channel
// that is used to find the region to perform convolution. Then it is assigned to a padded float array and returned
__global__ void sharpen(const float* input, const float* kernel, float* output, sharpen_metadata imgp) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    int d = threadIdx.z + blockIdx.z * blockDim.z;
    if (r < imgp.height && c < imgp.width && d < imgp.channels) {
        float value = 0;
        for (int i = -imgp.padding; i < imgp.padding + 1; i++) {
            for (int j = -imgp.padding; j < imgp.padding + 1; j++) {
                // idx = IMGIDX(r, c, d, padded_width, gridDim.z, padding, i, j);
                value += input[IMGIDX(r, c, d, imgp.padded_width, gridDim.z, imgp.padding, i, j)] *
                         kernel[(i + imgp.padding) * imgp.kernel_size + (j + imgp.padding)];
            }
        }
        //        printf("%f ", value > 255.0 ? 255.0 : value < 0.0 ? 0.0 : value);
        output[IMGIDX(r, c, d, imgp.padded_width, gridDim.z, imgp.padding, 0, 0)] =
            value > 255.0 ? 255.0 : value < 0.0 ? 0.0 : value;
    }
}

// used by both the cpu and the gpu
// takes an unsigned char img array and writes it to a padded float array,
// specified by the data in the sharpen_metadata struct
void padded_populate_img(float* dest_float_img, const unsigned char* img, sharpen_metadata imgp) {
    // populate the float_img with the values from img
    // casting to float allows for non integer kernels, if necessary
    // make sure the destination array is 0 before writing to it
    memset(dest_float_img, 0, imgp.num_float_bytes);
    int idx = 0, count = 0;
    for (int r = 0; r < imgp.height; r++) {
        for (int c = 0; c < imgp.width; c++) {
            for (int d = 0; d < imgp.channels; d++) {
                idx = IMGIDX(r, c, d, imgp.padded_width, imgp.channels, imgp.padding, 0, 0);
                dest_float_img[idx] = (float)img[count++];
            }
        }
    }
}

// Used by the gpu only, to take a padded float array and unpad while casting to unsigned char
// using information found in the sharpen metadata
void depopulate_img(unsigned char* dest_img, const float* float_img, sharpen_metadata imgp) {
    // make sure the destination array is 0 for num unsigned char bytes
    memset(dest_img, 0, imgp.num_pix * sizeof(unsigned char));
    int idx = 0, count = 0;
    for (int r = 0; r < imgp.height; r++) {
        for (int c = 0; c < imgp.width; c++) {
            for (int d = 0; d < imgp.channels; d++) {
                idx = IMGIDX(r, c, d, imgp.padded_width, imgp.channels, imgp.padding, 0, 0);
                dest_img[count++] = (unsigned char)float_img[idx];
            }
        }
    }
}

/********************** Wrapper Functions for Deblurring **********************/

// read in an unsigned char img array, multiply each pixel and its neighbors
// by a kernel, storing in a temp float variable, then clamping to unsigned char
// we do not need an intermediate float array like for the gpu.
unsigned char* run_on_cpu(const unsigned char* img, const float* kernel, sharpen_metadata imgp) {
    // even though there are no cuda calls, use the cuda event
    // timing to ensure timing schemes are equal between cpu and gpu
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed_time_ms = 0;

    cudaEventRecord(start);

    float* float_img = (float*)MALLOC_ERR(imgp.num_float_bytes);
    // printf("0x%" PRIXPTR ", 0x%" PRIXPTR "\n", (uintptr_t)float_img, (uintptr_t)(float_img + imgp.num_float_bytes));
    padded_populate_img(float_img, img, imgp);

    unsigned char* img_out = (unsigned char*)MALLOC_ERR(sizeof(unsigned char) * imgp.num_pix);
    memset(img_out, 255, sizeof(unsigned char) * imgp.num_pix); // shows if there are any errors for black backgrounds

    int idx = 0;
    int count = 0;
    float value = 0;
    for (int r = 0; r < imgp.height; r++) {
        for (int c = 0; c < imgp.width; c++) {
            for (int d = 0; d < imgp.channels; d++) {
                value = 0;
                for (int i = -imgp.padding; i < imgp.padding + 1; i++) {
                    for (int j = -imgp.padding; j < imgp.padding + 1; j++) {
                        idx = IMGIDX(r, c, d, imgp.padded_width, imgp.channels, imgp.padding, i, j);
                        value += float_img[idx] * kernel[(i + imgp.padding) * imgp.kernel_size + (j + imgp.padding)];
                    }
                }
                // if value is out of [0, 255], clamp it.
                img_out[count++] = value > 255.0 ? 255.0 : value < 0.0 ? 0.0 : (unsigned char)value;
                // printf("%d ", img_out[count - 1]);
            }
        }
    }

    free(float_img);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_ms, start, end);
    printf("Elapsed time for CPU: %f ms\n", elapsed_time_ms);
    return img_out;
}

// read in an unsigned char img array, convert it to a padded float array,
// allocate and copy the memory to the device and run the parallel kernel. Kernel returns clamped floats
// that just need to be cast.
unsigned char* run_on_gpu(const unsigned char* img, float* kernel, sharpen_metadata imgp) {
    // find out the cuda capabilities of the default device.
    // use the maxthreads per block to determine the num threads per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // get device properties
    printf("Max Threads Per Block for device 0: %d\n", prop.maxThreadsPerBlock);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed_time_ms = 0;

    cudaEventRecord(start);

    float* float_img = (float*)MALLOC_ERR(imgp.num_float_bytes);
    // float* float_img = NULL;
    // CUDA_CHECK_ERROR(cudaMallocHost((void**)&float_img, imgp.num_float_bytes));
    padded_populate_img(float_img, img, imgp);

    // get memory on the device for the input image and copy it over.
    // tried cudaHostAlloc for pinned host memory but that proved to be slower
    float* d_input_img = NULL;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_input_img, imgp.num_float_bytes));
    CUDA_CHECK_ERROR(cudaMemcpy(d_input_img, float_img, imgp.num_float_bytes, cudaMemcpyHostToDevice));

    // alloc the kernel on the device and copy it to the device
    float* d_kernel = NULL;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_kernel, sizeof(float) * imgp.kernel_size * imgp.kernel_size));
    CUDA_CHECK_ERROR(
        cudaMemcpy(d_kernel, kernel, sizeof(float) * imgp.kernel_size * imgp.kernel_size, cudaMemcpyHostToDevice));

    // alloc space for the result on the device
    float* d_output_img = NULL;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_output_img, imgp.num_float_bytes));

    // create 2d blocks and make a 3d grid of these blocks to cover the image
    int num_threads_per_block_dim = (int)sqrt(prop.maxThreadsPerBlock);
    int num_blocks_w = (imgp.padded_width + num_threads_per_block_dim - 1) / num_threads_per_block_dim;
    int num_blocks_h = (imgp.padded_height + num_threads_per_block_dim - 1) / num_threads_per_block_dim;
    printf("CUDA kernel launch with %d by %d blocks of %d by %d threads\n", num_blocks_w, num_blocks_h,
           num_threads_per_block_dim, num_threads_per_block_dim);
    dim3 block(num_threads_per_block_dim, num_threads_per_block_dim, 1);
    dim3 grid(num_blocks_h, num_blocks_w, imgp.channels);

    sharpen<<<grid, block>>>(d_input_img, d_kernel, d_output_img, imgp);

    // dump the result back into float_img
    CUDA_CHECK_ERROR(cudaMemcpy(float_img, d_output_img, imgp.num_float_bytes, cudaMemcpyDeviceToHost));

    unsigned char* img_out = (unsigned char*)MALLOC_ERR(sizeof(unsigned char) * imgp.num_pix);
    depopulate_img(img_out, float_img, imgp);

    free(float_img);
    cudaFree(d_input_img);
    cudaFree(d_kernel);
    cudaFree(d_output_img);

    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time_ms, start, end);
    printf("Elapsed time for GPU: %f ms\n", elapsed_time_ms);

    return img_out;
}

/***************************** Evaluation Metrics *****************************/

// calculates the luminance of a pixel
float luminance(unsigned char r, unsigned char g, unsigned char b) {
    return 0.2126 * (r / 255.0) + 0.7152 * (g / 255.0) + 0.0722 * (b / 255.0);
}

/*
 * Loop through each pixel in a row and compute the black and white pixel value of the next row and column
 * Store those values
 * Sum the total difference between those pixels and the current pixels to obtain a relative measure of sharpness
 * adapted from: https://www.mathworks.com/matlabcentral/fileexchange/32397-sharpness-estimation-from-image-gradients
 */
void evaluate_sharpness(const unsigned char* img, sharpen_metadata imgp, const char* img_name) {
    double dc2 = 0.0, dr2 = 0.0; // row-wise and column-wise differences, squared
    double dsharpness = 0.0;     // the norm of the row-wise and column-wise differences
    double rcur = 0.0, rnext = 0.0, ccur = 0.0, cnext = 0.0;
    double* tmp_row = (double*)MALLOC_ERR(sizeof(double) * imgp.width); // row-wise difference tmp storage
    int idx = 0;

    // populate the tmp array with the first row
    for (int c = 0; c < imgp.width; c++) {
        tmp_row[c] = luminance(img[c], img[c + 1], img[c + 2]);
    }

    for (int r = 0; r < imgp.height - 1; r++) {
        ccur = tmp_row[0]; // grab the 0th element here, as we do not set it inside the loop
        for (int c = 0; c < imgp.width - 1; c++) {
            rcur = tmp_row[c];
            idx = IMGIDX(r + 1, c, 0, imgp.width, imgp.channels, 0, 0, 0);
            rnext = luminance(img[idx], img[idx + 1], img[idx + 2]);
            dr2 = (rnext - rcur) * (rnext - rcur); // difference between rows, squared

            cnext = tmp_row[c + 1];
            dc2 = (cnext - ccur) * (cnext - ccur); // difference between two columns, squared

            dsharpness += sqrt(dc2 + dr2); // norm of the two differences squared

            tmp_row[c] = rnext;
            ccur = cnext;
        }
        idx = IMGIDX(r + 1, imgp.width - 1, 0, imgp.width, imgp.channels, 0, 0, 0);
        tmp_row[imgp.width - 1] = luminance(img[idx], img[idx + 1], img[idx + 2]); // compute last value for the tmp arr
    }
    double num_diff_pixels = (imgp.height - 1) * (imgp.width - 1);
    printf("Sharpness value of %-50s\t%0.6lf\n", img_name, dsharpness / num_diff_pixels);
}

// takes two image arrays and their names and computes the percent error (MSE), and prints it out to stdout
void compute_error(const unsigned char* img, const unsigned char* ref_img, int num_pix, const char* img_name,
                   const char* ref_img_name) {
    double error_val = 0.0;
    for (int i = 0; i < num_pix; i++) {
        error_val += pow((img[i] - ref_img[i]) / 255.0, 2);
    }
    printf("%% MSE: %-30s %-30s\t%0.4f%%\n", img_name, ref_img_name, 100 * error_val / num_pix);
}

/********************** Image and File Loading Functions **********************/

// Load in the image. Currently set to read in 3 channels for RGB.
// The argument filled by _dummy will be 4 for RGBA even if last arg is set to 3.
// Set last argument (imgp.channels) to 0 to read in all 4 channels, if available.
unsigned char* load_image_rgb(const char* img_file, sharpen_metadata* imgp) {
    sharpen_metadata* _imgp = imgp; // imgp might be null, but we assign it b/c if it is, we're about to overwrite _imgp
    if (imgp == NULL) _imgp = (sharpen_metadata*)MALLOC_ERR(sizeof(sharpen_metadata));

    unsigned char* img;
    int _dummy = 0;
    _imgp->width = -1, _imgp->height = -1, _imgp->channels = 3;
    img = stbi_load(img_file, &_imgp->width, &_imgp->height, &_dummy, _imgp->channels);
    if (_imgp->channels == 0) _imgp->channels = _dummy;
    if (img == NULL || _imgp->width <= 0 || _imgp->height <= 0 || _imgp->channels <= 0) {
        error_print(-1, 0, 6, "Error in loading \"%s\"\n", img_file);
    }
    printf("Loaded %s with width of %dpx, a height of %dpx and %d channels\n", img_file, _imgp->width, _imgp->height,
           _imgp->channels);

    _imgp->num_pix = _imgp->height * _imgp->width * _imgp->channels;

    _imgp->padding = _imgp->kernel_size / 2;
    _imgp->padded_width = _imgp->width + _imgp->padding * 2;
    _imgp->padded_height = _imgp->height + _imgp->padding * 2;
    _imgp->num_padded_pix = _imgp->padded_height * _imgp->padded_width * _imgp->channels;
    _imgp->num_float_bytes = sizeof(float) * _imgp->num_padded_pix;

    if (imgp == NULL) free(_imgp);

    return img;
}

float* load_kernel(const char* kernel_file, sharpen_metadata* imgp) {
    float* kernel;

    FILE* fptr = fopen(kernel_file, "r");
    if (!fptr) {
        error_print(-1, 1, 2, "Error opening deconvolution kernel file %s: ", kernel_file);
    }
    else {
        fscanf(fptr, "%d", &imgp->kernel_size);
        if (imgp->kernel_size % 2 != 1 || imgp->kernel_size < 0) {
            error_print(-1, 0, 3, "Invalid kernel size. Kernel size must be a positive, odd integer: %d\n",
                        imgp->kernel_size);
        }
        printf("Kernel Size:\t%d by %d\n", imgp->kernel_size, imgp->kernel_size);
        kernel = (float*)malloc(sizeof(float) * imgp->kernel_size * imgp->kernel_size);
        for (int i = 0; i < imgp->kernel_size; i++) {
            for (int j = 0; j < imgp->kernel_size; j++) {
                error_print(fscanf(fptr, "%f", &kernel[i * imgp->kernel_size + j]), 1, 4,
                            "Error during fscanf (if error is \"Success\" then EOF may have been reached prematurely. "
                            "Please ensure deconvolution kernel file is valid");
            }
        }
    }
    error_print(fclose(fptr), 1, 5, "Error while closing deconvolution kernel file: %s\n", kernel_file);

    if (imgp->kernel_size <= 21) {
        printf("Deconvolution kernel:\n");
        for (int i = 0; i < imgp->kernel_size; i++) {
            for (int j = 0; j < imgp->kernel_size; j++) {
                printf("%6.2f\t", kernel[i * imgp->kernel_size + j]);
            }
            printf("\n");
        }
    }
    else {
        printf("Deconvolution kernel is too large to be displayed. Continuing.");
    }
    return kernel;
}

/************************* Argument Parsing Functions *************************/

typedef struct arg_options {
    const char* kernel_file;
    const char* input_img_file;
    const char* ref_img_file;
    const char* output_img_prefix;
    const char* device_choice;
} arg_options;

void parse_args(arg_options* opt, int ac, const char** av) {
    poptContext optCon;
    struct poptOption optionsTable[] = {
        {"kernel", 'k', POPT_ARG_STRING, &opt->kernel_file, 0,
         "Space separated file designating the deconvolution kernel. First line is a single INT that describes the "
         "size of the kernel. Subsequent lines describe the actual kernel.",
         "kernel_file"},
        {"input", 'i', POPT_ARG_STRING, &opt->input_img_file, 0, "Image to deblur", "blurred_image_file"},
        {"reference", 'r', POPT_ARG_STRING, &opt->ref_img_file, 0,
         "Non-blurred, sharp image. Used to assess sharpness. If not provided, error will not be computed",
         "sharp_image_file"},
        {"output_prefix", 'o', POPT_ARG_STRING, &opt->output_img_prefix, 0,
         "Output image prefix. Will be followed by _cpu.png/_gpu.png. Default: \"output\"", "prefix_string"},
        {"device", 'd', POPT_ARG_STRING, &opt->device_choice, 0, "Specify to run on gpu, cpu, or both. Default: both",
         "gpu|cpu|both"},
        POPT_AUTOHELP POPT_TABLEEND};
    optCon = poptGetContext(NULL, ac, av, optionsTable, 0);
    char rc = poptGetNextOpt(optCon);

    if (ac < 2) {

        printf("No arguments specified, defaulting to:\n");
    }
    else {
        printf("Specified Arguments:\n");
    }
    printf("Kernel File:\t%s\n"
           "Input Image:\t%s\n"
           "Ref Image:\t%s\n"
           "Output Prefix:\t%s\n"
           "Device Name:\t%s\n",
           opt->kernel_file, opt->input_img_file, opt->ref_img_file, opt->output_img_prefix, opt->device_choice);
}

int get_device(const char* choice) {
    // assign a device
    if (strcmp(choice, "gpu") == 0) {
        return 0;
    }
    else if (strcmp(choice, "cpu") == 0) {
        return 1;
    }
    else if (strcmp(choice, "both") == 0) {
        return 2;
    }
    else {
        error_print(-1, 0, 1, "Invalid device specified: %s\nDevice should be chosen from [gpu|cpu|both]\n");
        return -1;
    }
}

/************************************ Main ************************************/

int main(int argc, const char** argv) {
    arg_options opt;
    opt.kernel_file = "masks/kernel7x7.txt";
    opt.input_img_file = "images/nasa_earth_blurry.png";
    opt.ref_img_file = "images/nasa1R3107_1000x1000.jpg";
    opt.output_img_prefix = "output/output";
    opt.device_choice = "both";
    parse_args(&opt, argc, argv);

    int device = get_device(opt.device_choice); // 0 for gpu, 1 for cpu, 2 for both

    // load in the deconvolution kernel
    sharpen_metadata imgp;
    float* kernel = load_kernel(opt.kernel_file, &imgp); // must load kernel first because it sets imgp->kernel_size
    unsigned char* img = load_image_rgb(opt.input_img_file, &imgp); // this sets the remaining params of imgp

    unsigned char* cpu_output = NULL;
    unsigned char* gpu_output = NULL;
    char gpu_output_filename[strlen(opt.output_img_prefix) + MAXFILENAMELEN];
    char cpu_output_filename[strlen(opt.output_img_prefix) + MAXFILENAMELEN];
    
    if (device < 0 || device > 2) {
        error_print(-1, 0, 255, "Something really wrong happened. device choice is: %s\n", opt.device_choice);
    }
    if (device == 0 || device == 2) {
        gpu_output = run_on_gpu(img, kernel, imgp);
        gpu_output_filename[0] = '\0';
        strcat(gpu_output_filename, opt.output_img_prefix);
        strcat(gpu_output_filename, "_gpu.jpg");
        stbi_write_jpg(gpu_output_filename, imgp.width, imgp.height, imgp.channels, gpu_output, 100);
    }
    if (device == 1 || device == 2) {
        cpu_output = run_on_cpu(img, kernel, imgp);
        cpu_output_filename[0] = '\0';
        strcat(cpu_output_filename, opt.output_img_prefix);
        strcat(cpu_output_filename, "_cpu.jpg");
        stbi_write_jpg(cpu_output_filename, imgp.width, imgp.height, imgp.channels, cpu_output, 100);
    }

    // now we need to calculate sharpness and compute error
    // first, we load in ref_imgp
    sharpen_metadata ref_imgp;
    unsigned char* ref_img = NULL;
    if (opt.ref_img_file != NULL) {
        ref_img = load_image_rgb(opt.ref_img_file, &ref_imgp);

        int ref_valid = 1;
        if (ref_imgp.width != imgp.width || ref_imgp.height != imgp.height || ref_imgp.channels != imgp.channels) {
            fprintf(stderr, "Reference Image %s dimensions do not match %s.\nDefaulting to only evaluating sharpness",
                    opt.ref_img_file, opt.input_img_file);
            ref_valid = 0;
        }

        printf("\nEffectiveness (Sharpness) Metrics (Higher is Estimated to be Sharper):\n");
        if (ref_valid) evaluate_sharpness(ref_img, ref_imgp, opt.ref_img_file);
        evaluate_sharpness(img, imgp, opt.input_img_file);
        if (gpu_output != NULL) evaluate_sharpness(gpu_output, imgp, gpu_output_filename);
        if (cpu_output != NULL) evaluate_sharpness(cpu_output, imgp, cpu_output_filename);

        // compute error but only if the reference image is valid
        if (ref_valid) {
            printf("\nError Computations (Percent MSE):\n");
            printf("       %-30s %-30s\n", "Image", "Reference");
            compute_error(img, ref_img, imgp.num_pix, opt.input_img_file, opt.ref_img_file);
            if (gpu_output != NULL)
                compute_error(gpu_output, ref_img, imgp.num_pix, gpu_output_filename, opt.ref_img_file);
            if (cpu_output != NULL)
                compute_error(cpu_output, ref_img, imgp.num_pix, cpu_output_filename, opt.ref_img_file);
        }
    }
    else {
        printf("\nNo reference image provided. Only computing sharpness.\n");
        printf("Effectiveness (Sharpness) Metrics (Higher is Estimated to be Sharper):\n");
        evaluate_sharpness(img, imgp, opt.input_img_file);
        if (gpu_output != NULL) evaluate_sharpness(gpu_output, imgp, gpu_output_filename);
        if (cpu_output != NULL) evaluate_sharpness(cpu_output, imgp, cpu_output_filename);
    }

    stbi_image_free(img);
    free(kernel);
    if (gpu_output != NULL) free(gpu_output);
    if (cpu_output != NULL) free(cpu_output);
    if (ref_img != NULL) stbi_image_free(ref_img);
    return 0;
}
