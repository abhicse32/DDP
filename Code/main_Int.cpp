#include <iostream>
#include "AVL_Sequential_Int.cpp"
#include <chrono>
#include <sys/time.h>
#define NUM 1E6
using namespace std;
int main(int argc, char* argv[]){
	int i,j,size,n;
	double time_taken=0;

	for(int j=0;j<10;++j){
		Tree<int>* tree= new Tree<int>();
		int max_search= atoi(argv[2]);
		tree->insert_nodes(argv[1],max_search);
		size=(tree->size)-max_search;
		// generating random elements to perform query
		int* temp_arr= new int[max_search];
		// for(int j=0;j<10;++j){
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

		auto start = std::chrono::high_resolution_clock::now();
		int half= (max_search>>1);
		for(int i=0;i<max_search;++i){
			// if(i<=half)
				tree->search(temp_arr[i]);
			// else
			// 	tree->delete_(temp_arr[i]);
		}

		auto end= std::chrono::high_resolution_clock::now();
		time_taken+=(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/NUM);
		cout << time_taken<< endl;
		delete temp_arr;
		delete tree->arr;
		delete tree;
	}
	cout << (time_taken)/10 << "ms\n";
}