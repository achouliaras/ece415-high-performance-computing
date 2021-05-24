/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gputimer.h"
unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

//#define CPU_compute

__device__ __constant__ double d_Filter[2*256+1]={0};

////////////////////////////////////////////////////////////////////////////////
// GPU row convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convRowGPU( double *d_Dst, double *d_Src,
                       int imageW, int imageH, int filterR ){
    int ix= blockDim.x * blockIdx.x + threadIdx.x +filterR;
    int iy= blockDim.y * blockIdx.y + threadIdx.y +filterR;
    int ik;
    double sum=0;
    for (ik = -filterR; ik <= filterR; ik++) {
        int d = ix + ik;

        sum += d_Src[iy * (imageW+ 2*filterR) + d] * d_Filter[filterR - ik];
    }
    d_Dst[iy * (imageW+ 2*filterR) + ix] = sum;
}
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}
////////////////////////////////////////////////////////////////////////////////
// GPU column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convColGPU( double *d_Dst, double *d_Src,
                       int imageW, int imageH, int filterR ){
    int ix= blockDim.x * blockIdx.x + threadIdx.x +filterR;
    int iy= blockDim.y * blockIdx.y + threadIdx.y +filterR;
    int ik;
    double sum = 0;
    for (ik = -filterR; ik <= filterR; ik++) {
      int d = iy + ik;

      sum += d_Src[d * (imageW+ 2*filterR) + ix] * d_Filter[filterR - ik];
    }
    d_Dst[iy * (imageW+ 2*filterR) + ix] = sum;
}
////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    double
    //*d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    unsigned int imageW;
    unsigned int imageH;
    unsigned int i;
    int rval;

    #ifdef CPU_compute
    struct timespec  tv1, tv2;
    #endif

	printf("Enter filter radius : ");
	rval=scanf(" %d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    rval=scanf(" %d", &imageW);
    if (rval== -1){
        printf("ERROR scanf");
    }
    imageH = imageW;
    unsigned int block_size=(imageW<=32) ? imageW : 32;
    unsigned int gridsize=(imageW)/block_size;
    if (imageW % block_size != 0)
        gridsize++;
    dim3 dimGrid( gridsize, gridsize);
    dim3 dimBlock( block_size, block_size);   // 64x64 den douleuei

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

    // if any memory allocation failed, report an error message
    if(h_Filter == 0 || h_Input == 0 || h_Buffer == 0 || h_OutputCPU == 0 || h_OutputGPU == 0){
        printf("CPU couldn't allocate memory\n");
        return 1;
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }

    // GPU code section starts here
    GpuTimer timer;
    //Memory Allocation
    //cudaMalloc((void**)&d_Filter   , FILTER_LENGTH * sizeof(double));
    cudaMalloc((void**)&d_Input    , (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));
    cudaMalloc((void**)&d_Buffer   , (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));
    cudaMalloc((void**)&d_OutputGPU, (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));

    // if any memory allocation failed, report an error message
    //d_Filter == 0 ||
    if( d_Input == 0 || d_Buffer == 0 || d_OutputGPU == 0 ){
        printf("GPU couldn't allocate memory\n");
        return 1;
    }
    //Memory Initialization
    //cudaMemset(d_Filter   ,0, FILTER_LENGTH * sizeof(double));
    cudaMemset(d_Input    ,0, (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));
    cudaMemset(d_Buffer   ,0, (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));
    cudaMemset(d_OutputGPU,0, (imageW + 2*filter_radius)* (imageH + 2*filter_radius) * sizeof(double));

    printf("GPU computation...\n");
    timer.Start();                       // START GPU TIMER

    cudaMemcpyToSymbol(d_Filter,h_Filter, FILTER_LENGTH * sizeof(double), 0 , cudaMemcpyHostToDevice);
    for(i=0;i<imageH;i++){
        cudaMemcpy(d_Input+((i+filter_radius)*(imageW+2*filter_radius)+filter_radius) ,h_Input+(i*imageW) ,\
                                                        imageW * sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_Buffer+((i+filter_radius)*(imageW+2*filter_radius)+filter_radius) ,h_Buffer+(i*imageW),\
                                                        imageW * sizeof(double), cudaMemcpyHostToDevice);
    }

    //timer.Start();                       // START GPU TIMER
    // GPU convolution kata grammes
    convRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, imageW, imageH, filter_radius);
    // convolution kata sthles
    convColGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, imageW, imageH, filter_radius);
    //timer.Stop();                        // FINISH GPU TIMER

    for(i=0;i<imageH;i++){
        cudaMemcpy(h_OutputGPU+(i*imageW), d_OutputGPU+((i+filter_radius)*(imageW+2*filter_radius)+filter_radius) ,\
                                                        imageW * sizeof(double), cudaMemcpyDeviceToHost);
     }
    timer.Stop();                        // FINISH GPU TIMER

    #ifdef CPU_compute

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

    //Standard accuracy check
    rval = 0;
    for (i = 0; i < imageW * imageH; i++) {
        if( ABS(h_OutputGPU[i]-h_OutputCPU[i]) > accuracy){
            printf("The two images differ...\n");
            printf("%d ->%f \t ->%f\n",i,h_OutputGPU[i],h_OutputCPU[i]);
            rval = 1;
            break;
        }
    }
    if(rval == 0){
        printf("The two images are the SAME!!!\n");
    }

    // find max accuracy for given filter radius
    /*
    double acc=0;
    for( acc=0;acc<10; acc=acc+0.000001){
        rval = 0;
        for (i = 0; i < imageW * imageH; i++) {
            if( ABS(h_OutputGPU[i]-h_OutputCPU[i]) > acc){
                //printf("The two images differ...%lf\n",acc);
                rval = 1;
                break;
            }
        }
        if(rval == 0){
            printf("The two images are the SAME!!! %lf\n",acc);
            break;
        }
    }
    */
    #endif

    printf("GPU Time elapsed = %-10g s\n", timer.Elapsed()/1000.0);
    #ifdef CPU_compute
    printf("CPU time elapsed = %-10g s\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
    #endif
    // free all the allocated memory
    //cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);
    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaDeviceReset();

    return 0;
}
