#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <fstream>
using namespace std;

#define pi 3.1415926
#define lambda 6.7e-7  // wavelength
#define X 0.05      // source-detector distance
#define gamma 0.999   // learning rate

// M = 256
// num of detectors is 2*M = 512
// N is the estimate of number of photons arriving at each detector

// total num of samples = nthreads * sampleperthread
// event is a flattened array of size
// (nthreads, sampleperthread, 2)) --> (nthreads*sampleperthread*2)

__global__
void sample(float* event, int sampleperthread, float d, float a, int M)
{                                
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;

  curandStateXORWOW my_curandstate;
  curand_init((unsigned long long) clock() + idx, 0, 0, &my_curandstate);

  for (int i = 0; i < 2*sampleperthread; i=i+2){
    // generate random ys between [(d+a)/2, (d-a)/2]
    float r1, r2, y;    // the y coordiante where the photon leaves the source
    r1 = curand_uniform(&my_curandstate);
    y = (2*r1 - 1) * a / 2;
    r2 = curand_uniform(&my_curandstate);
    if (r2 > 0.5)
      y = y + d/2;
    else
      y = y - d/2;

    // generate random betas between [-pi/2, pi/2]
    float r3, beta; 
    r3 = curand_uniform(&my_curandstate);
    beta = (2*r3 - 1) * pi /2;
    
    float z, theta, s, phi;
    z = y / X;

    // calculate the angular coordiante of the detector being hit
    theta = asin(z * pow(cosf(beta), 2) + sinf(beta) * sqrtf(1 - pow(z * cosf(beta), 2)));
    s = X * sqrt(1 - 2 * z * sin(beta) + z*z);
    phi = 2 * pi * s / lambda;

    //convert the angular coordiante to detector index [0, 2*M-1]
    float dtheta = (pi / 3) / M;   // detection window of each detector
    float idx_det = theta / dtheta + M;

    event[sampleperthread*idx*2 + i] = phi;
    event[sampleperthread*idx*2 + i + 1] = idx_det; 

  }
}

__global__
void sort(float* event, float* det, int* len, int M, int N, int sum)
{
	for (int i=0; i<M; i++){
		len[i] = 0;
		}
	for (int i=0; i <= sum; i=i+2){
		int det_idx = event[i+1];
		if (det_idx >= 0 && det_idx < 2*M){
			det[det_idx * N + len[det_idx]] = event[i];
			len[det_idx] += 1;
			}
		}
}


__global__
void detect(float* det, float* int_vec, int* len, int* count, int M, int N)
{
	// nthread index == detector index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	curandStateXORWOW my_curandstate;
	curand_init((unsigned long long) clock() + idx, 0, 0, &my_curandstate);
	
	// initialise the internal vectors [2*M, 2]
	for (int i=0; i < 2*M*2; i++){
		int_vec[i] = 1.0 / sqrtf(2);
		}
	
	// initialise the detector counts
	for (int i=0; i<2*M; i++){
		count[i]=0;}
		
	// detection
	for (int i=0; i<=len[idx]; i++){
		//update rule
		int_vec[idx*2] = gamma * int_vec[idx*2] + (1-gamma) * cosf(det[N*idx+i]);
		int_vec[idx*2+1] = gamma * int_vec[idx*2+1] + (1-gamma) * sinf(det[N*idx+i]);
		//threshold
		float r = curand_uniform(&my_curandstate);
		float threshold = int_vec[idx*2]*int_vec[idx*2] + int_vec[idx*2+1]*int_vec[idx*2+1];
		
		if (threshold > r){
			count[idx] += 1;
			}
		}
}


int main(void)
{
  int nthreads = 1<<22;
  int sampleperthread = 8;
  int sum = nthreads * sampleperthread;
  int M = 256;
  int N = 1<<16;
// allocate memory for event on device
  float* event;
  size_t size = nthreads * sampleperthread * sizeof(float);
  cudaMalloc(&event, 2*size);
// allocate memory for det on device
  float* det;
  cudaMalloc(&det, 2*M*N*sizeof(float));
// allocate memory for len on device
  int* len;
  cudaMalloc(&len, 2*M*sizeof(int));
  
  float* int_vec;
  cudaMalloc(&int_vec, 2*M*2*sizeof(float));
  
  int* count;
  cudaMalloc(&count, M*2*sizeof(int));

  float* h_event = (float*)malloc(2*size);
  float* h_det = (float*)malloc(2*M*N*sizeof(float));
  int* h_count = (int*)malloc(M*2*sizeof(int));

  int blockSize = 32;
  int numBlocks = (nthreads + blockSize - 1) / blockSize;

  float d = 5 * lambda; 
  float a = lambda;

  sample<<<numBlocks, blockSize>>>(event, sampleperthread, d, a, M);

  sort<<<1, 1>>>(event, det, len, M, N, sum);
  
  int num = (2*M + blockSize - 1) / blockSize;
  
  detect<<<num, blockSize>>>(det, int_vec, len, count, M, N);

  cudaDeviceSynchronize();
  
  //cudaMemcpy(h_event, event, 2*size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_det, det, 2*M*N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_count, count, M*2*sizeof(int), cudaMemcpyDeviceToHost);
  
  
  ofstream MyFile("count.txt");
  for(int i = 0; i < 2*M; i++){
	  MyFile << h_count[i] << "\n" ;}

	MyFile.close();

  cudaFree(event);
  cudaFree(det);
  cudaFree(len);
  cudaFree(count);

  return 0;
}
