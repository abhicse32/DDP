#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "AVL_Concurrent_Lib.cu"
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int getGlobalId(){
	int blockId= gridDim.x * gridDim.y * blockIdx.z + 
		     gridDim.x * blockIdx.y + blockIdx.x;
	int threadId= blockId*(blockDim.x * blockDim.y * blockDim.z)+ 
		      blockDim.x * blockDim.y * threadIdx.z + 
			blockDim.x * threadIdx.y + threadIdx.x;
	return threadId;
}

__global__ void start_kernel(Tree<int>* tree,int* d_query, int* max_query,
							int* three_4th, int* half, int* mutex){
	int threadId= getGlobalId();
	for(int i=threadId;i< *max_query;i+=512){
		if(i<*max_query){
			if(i<*half){
				tree->search_2(d_query[i]);
			}
			else{
				bool leave_loop=false;
				while(!leave_loop){
					if(atomicExch(mutex,1)==0){
						if(i<*three_4th)
							tree->delete_(d_query[i]);
						else tree->insert(d_query[i]);
						leave_loop=true;
						atomicExch(mutex,0);
					}
				}
			}
		}
	}
	__syncthreads();
}

int* generate_query(Tree<int>* tree,int size,int max_search){
	int* temp_arr= new int[max_search];
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    srand((time_t)ts.tv_nsec);
    for(int i=0;i<max_search;++i){
        if(i%10==0){
        	int index= 2+rand()%(size-2);
	   	 	temp_arr[i]= tree->arr[index].key;
        }
        else
	    	temp_arr[i]= rand()%max_search;       
    }
    return temp_arr;
}

int main(int argc,char* argv[]){
	Tree<int>* tree= new Tree<int>();
	int max_search= atoi(argv[2]);
	tree->insert_nodes(argv[1],max_search);
 	
	int half= (max_search>>1);
	int three_4th=((3*max_search)>>2);
	int size_2= tree->size;
	// for(int i=0;i<5;++i){
 	int *d_query;
	int* d_max_query;
	Node<int>* d_arr;
	Tree<int>* d_tree;
   	int *d_half;
   	int *d_three_4th;
   	int *d_mutex;
    int*temp_arr= generate_query(tree,(size_2 - max_search),max_search);
    
	gpuErrchk(cudaMalloc((void**)&d_arr,sizeof(Node<int>)*size_2));
	gpuErrchk(cudaMalloc((void**)&d_tree,sizeof(Tree<int>)));
	gpuErrchk(cudaMalloc((void**)&d_query,sizeof(int)*max_search));
	gpuErrchk(cudaMalloc((void**)&d_max_query,sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_half,sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutex,sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_three_4th,sizeof(int)));

	gpuErrchk(cudaMemcpy(d_tree,tree,sizeof(Tree<int>),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_arr,tree->arr,sizeof(Node<int>)*size_2,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&(d_tree->arr),&d_arr,sizeof(Node<int>*),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_query,temp_arr,sizeof(int)*max_search,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_max_query,&max_search,sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_half,&half,sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_three_4th,&three_4th,sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_mutex,0,sizeof(int)));

	dim3 block_dim(8,8,8);
    dim3 grid_dim(8,8,8); 

	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	start_kernel<<<grid_dim,block_dim>>>(d_tree,d_query,d_max_query, 
										d_three_4th,d_half,d_mutex);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// time_taken+=(elapsed_time/1000.0);

	cudaFree(d_arr);
	cudaFree(d_query);
	cudaFree(d_tree);
	cudaFree(d_half);
	cudaFree(d_three_4th);
	cudaFree(d_max_query);
	cudaFree(d_mutex);
}
