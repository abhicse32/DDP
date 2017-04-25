#include <iostream>
#include "avl_logical_str.cpp"
#include <chrono>
#include <sys/time.h>
#define NUM 1E6
using namespace std;

char** generate_queries(Tree<char>* tree,int size, int MAX_SEARCH){
        char** query_arr=new char*[MAX_SEARCH];
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC,&ts);
        srand((time_t)ts.tv_nsec);

        int max_query_len=20;
        char str[max_query_len];
        for(int i=0;i<MAX_SEARCH;++i){
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
    return query_arr;	
}

int main(int argc, char* argv[]){
	Tree<char>*tree= new Tree<char>();
	int max_search= atoi(argv[2]);
	tree->insert_nodes(argv[1],max_search);
	int i,j,size,n;
	double time_taken=0;

	for(int j=0;j<10;++j){
		Tree<char>* tree= new Tree<char>();
		int max_search= atoi(argv[2]);
		tree->insert_nodes(argv[1],max_search);
		size=(tree->size)-max_search;
		// generating random elements to perform query
		char** query_arr=generate_queries(tree,size,max_search); 

		auto start = std::chrono::high_resolution_clock::now();
		int half= (max_search>>1);
		int third_4th= int((3.0*max_search)/4);
		for(int i=0;i<max_search;++i){
			if(i<=half)
				tree->search(query_arr[i]);
			else if(i<=third_4th)
				tree->insert(query_arr[i]);
			else tree->delete_(query_arr[i]);
		}

		auto end= std::chrono::high_resolution_clock::now();
		time_taken+=(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/NUM);
		cout << time_taken<< endl;
		delete query_arr;
		delete tree->arr;
		delete tree;
	}
	cout << (time_taken)/10 << "ms\n";
}