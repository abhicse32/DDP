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
#include "AVL_Concurrent_Str_Lib.cu"
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

__global__ void start_kernel(Tree<char>* tree,char** d_query, int* max,
							int* half,int* third_4th, int* mutex){
	int threadId= getGlobalId();
	for(int i=threadId;i< *max;i+=512){
		if(i<*max){
			if(i< *half)
				tree->search_2(d_query[i]);
			else{
				bool leave_loop=false;
				while(!leave_loop){
					if(atomicExch(mutex,1)==0){
						tree->insert(d_query[i]);
						leave_loop=true;
						atomicExch(mutex,0);
					}
				}
			}
		}
	}
	__syncthreads();
}
char** generate_queries(Tree<char>* tree,int size, int max_search){
        char** query_arr=new char*[max_search];
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC,&ts);
        srand((time_t)ts.tv_nsec);

        int max_query_len=20;
        char str[max_query_len];
        for(int i=0;i<max_search;++i){
		// 10% queries will consists of the keys present in the tree
	        if(i%10==0){
		    int index= 2+rand()%(size-2);
	            query_arr[i]= strdup(tree->arr[index].key);
	        }else{
		    for(int j=0;j<max_query_len;++j)
			    str[j]='a'+rand()%26;
		    str[max_query_len-1]='\0';
		    query_arr[i]= strdup(str);	 
		} 
    }
    // completed search-query generation   
    return query_arr;	
}

char** copy_query_insert_data(char** query_arr, int max_search){
       char** query_device_mem;
       cudaMalloc((void**)&query_device_mem,sizeof(char*)*max_search);
      //deep copying query-data
       char** d_temp_data=(char**)malloc(sizeof(char*)*max_search);
       for(int i=0;i<max_search; ++i){
		   int len= strlen(query_arr[i])+1;
	           cudaMalloc(&(d_temp_data[i]),len*sizeof(char));
		   cudaMemcpy(d_temp_data[i],query_arr[i],len*sizeof(char),cudaMemcpyHostToDevice);
		   cudaMemcpy(query_device_mem+i,&(d_temp_data[i]),sizeof(char*),cudaMemcpyHostToDevice);    
       }
       return query_device_mem;
}

Tree<char>* temp_copy2(Tree<char>* h_tree, int h_size){
	Tree<char>* d_tree;
	Node<char> *d_arr;
	cudaMalloc(&d_arr,sizeof(Node<char>)*h_size);
	cudaMalloc(&d_tree,sizeof(Tree<char>));
	cudaMemcpy(d_arr,h_tree->arr,h_size*sizeof(Node<char>),cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree,h_tree,sizeof(Tree<char>),cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_tree->arr),&d_arr,sizeof(Node<char>*),cudaMemcpyHostToDevice);
	return d_tree;
}	

int main(int argc,char* argv[]){
	Tree<char>* tree= new Tree<char>();
	int max_search= atoi(argv[2]);
	tree->insert_nodes(argv[1],max_search);
 	// tree->inorder(tree->root);
	int half= (max_search>>1);
	int third_4th= int((3.0*max_search)/4);
	int size_2= tree->size;
	int* d_mutex;

	char** query_arr= generate_queries(tree,(size_2 - max_search),max_search);
 
    int *d_third_4th, *d_half;
    Tree<char>* d_tree;
    int* d_max;
  
      // allocates memory on device
    cudaMalloc((void**)&d_third_4th,sizeof(int));
    cudaMalloc((void**)&d_half,sizeof(int));
    cudaMalloc((void**)&d_max,sizeof(int));
    cudaMalloc((void**)&d_mutex,sizeof(int));
    char** d_query= copy_query_insert_data(query_arr,max_search);
    cudaMemcpy(d_max,&max_search,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_third_4th,&third_4th,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_half,&half,sizeof(int),cudaMemcpyHostToDevice); 
    cudaMemset(&d_mutex,0,sizeof(int));
    d_tree=temp_copy2(tree,size_2);

   // size_t buff_size;
   // cudaDeviceGetLimit(&buff_size,cudaLimitPrintfFifoSize);
   // buff_size= buff_size * 100;
   // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, buff_size);
   dim3 block_dim(8,8,8);
   dim3 grid_dim(1,1,1);

	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	start_kernel<<<grid_dim,block_dim>>>(d_tree,d_query,d_max,
										d_half,d_third_4th,d_mutex);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// time_taken+=(elapsed_time/1000.0);
	printf("GPU time elapsed: %f ms\n",elapsed_time/1000.0);
}