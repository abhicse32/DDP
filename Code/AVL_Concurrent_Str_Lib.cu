#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
using namespace std;


/*structure of the nodes of the tree*/
__host__ __device__ int strcmp_(char* str1,char* str2){
	const unsigned char* ptr1= (const unsigned char*)str1;
    const unsigned char* ptr2= (const unsigned char*)str2;
	while(ptr1 && *ptr1 == *ptr2) 
		++ptr1, ++ptr2;
	return *ptr1 - *ptr2;
}

template<class T>
class Node{
	public:
		int parent;
		T key[80];
		int height;
		int left;
		int right;
		int succ,pred;

		
		__host__ __device__ Node(T* key_){
			strcpy(key,key_);
			parent=height=left=right=INT_MIN;
			succ=pred=INT_MIN;
		}

		__host__ __device__ Node(){
			new(this)Node("");
		}

		friend ostream &operator<<(ostream &os, Node& node){
			os << node.key;
			return os;
		}

		__host__ __device__ bool operator<(Node &node){
			return strcmp_(key,node.key)<0;
		}

		__host__ __device__ bool operator==(Node& node){
			return strcmp_(key,node.key)==0;
		}
		__host__ __device__ bool operator>(Node &node){
			return strcmp_(key,node.key)>0;
		}
};

/*Definition of the tree*/
template<class T>
class Tree{
	public:
		int root,next_index;
		int size;
		/*container of the tree*/
		//node at index 0 is sentinel node with key INT_MIN
		// initialized root with key INT_MAX and index=1
		Node<T>* arr;

		__host__ __device__ Tree(int size_):root(1),next_index(2),size(size_){
			arr= new Node<T>[size];
			char str_[80];
			for(int i=0;i<79;++i)
				str_[i]=(char)127;
			strcpy(arr[root].key,str_);
			arr[root].parent = 0;
			arr[0].right= root;
			arr[root].pred= 0;
			arr[0].succ= root;
			arr[root].height=0;
		};

		__host__ __device__ Tree(){}
		 void inorder(int);
		 void preorder(int);
		 void postorder(int);
		__host__ __device__  int search(Node<T>&);
		__host__ __device__  int search(T*);	
	    __host__ __device__  int height(int);
		__host__ __device__  void insert(T*);
		__host__ __device__  void delete_(T*);
		__host__ __device__  void delete_rebalance(int);
		__host__ __device__  void init_node(Node<T>&, int, int, int, int);
		__host__ __device__  int tri_node_restructure(int, int, int);
		__host__ __device__  void recompute_height(int);
		__host__ __device__  int taller_child(int);
		__host__ __device__ bool search_2(T*);
		void insert_nodes(char*,int);

};

template<class T> 
void Tree<T>::inorder(int index){
	if(index==INT_MIN || strcmp_(arr[index].key,"")==0)
		return;
	inorder(arr[index].left);
        cout<< arr[index] <<"\n";
	inorder(arr[index].right);
}

template<class T>
void Tree<T>::preorder(int index){
	if(index==INT_MIN || strcmp_(arr[index],"")==0)
		return;
	cout << arr[index] <<" ";
	preorder(arr[index].left);
	preorder(arr[index].right);
}

template<class T> 
void Tree<T>::postorder(int index){
	if(index==INT_MIN || strcmp_(arr[index].key,"")==0)
		return;
	postorder(arr[index].left);
	postorder(arr[index].right);
	cout << arr[index] <<" ";
}


template<typename T>  __host__ __device__ 
int Tree<T>::search(Node<T> &node){
    int temp=root;
    while(temp!=INT_MIN){
        if(arr[temp]==node)
            return temp;
        int child= (arr[temp]<node? arr[temp].right: 
        	 			arr[temp].left);
        if(child==INT_MIN)
        	return temp;
        temp= child;
    }	
    return temp;
}

template<typename T> __host__ __device__ 
int Tree<T>::search(T* key_){
	int temp= root;
	while(temp!=INT_MIN){
		if(arr[temp].key == key_)
			return temp;
		int child= (arr[temp].key <key_?arr[temp].right
								:arr[temp].left);
		if(child==INT_MIN)
			return temp;
		temp=child;
	}
	return temp;
}


template<typename T>  __host__ __device__ 
void Tree<T>::recompute_height(int x){
	while(x!=root){
		arr[x].height = max(height(arr[x].right), height(arr[x].left));
		x= arr[x].parent;
	}
}

template<typename T> __host__ __device__ 
int Tree<T>::height(int index){
	if(index==INT_MIN)
		return 0;
	return arr[index].height+1;
}
template<typename T> __host__ __device__ 
int Tree<T>::tri_node_restructure(int x, int y, int z){
	/*
		x= parent(y)
		y= parent(z)
	*/
	bool z_is_left_child= (arr[y].left ==z);
	bool y_is_left_child = (arr[x].left== y);
	int a=INT_MIN,b=INT_MIN,c=INT_MIN;
	int t0=INT_MIN,t1=INT_MIN,t2=INT_MIN,t3=INT_MIN;
	if(z_is_left_child && y_is_left_child){
		a= z; b = y; c= x;
		t0 = arr[z].left; t1= arr[z].right;
		t2= arr[y].right; t3= arr[x].right;
		// printf("first if: %d %d %d %d\n",t0,t1,t2,t3);
	}else if(!z_is_left_child && y_is_left_child){
		a= y; b=z; c= x;
		t0= arr[y].left; t1= arr[z].left;
		t2= arr[z].right; t3= arr[x].right;
		// printf("second if: %d %d %d %d\n",t0,t1,t2,t3);
	}else if(z_is_left_child && !y_is_left_child){
		a=x; c= y; b=z;
		t0= arr[x].left; t1= arr[z].left;
		t2= arr[z].right; t3= arr[y].right;
		// printf("third if: %d %d %d %d\n",t0,t1,t2,t3);
	}else{
		a=x; b=y; c=z;
		t0= arr[x].left; t1= arr[y].left;
		t2= arr[z].left; t3= arr[z].right;
		// printf("fourth if:%d %d %d %d %d\n",arr[z].left,t0,t1,t2,t3);
	}
	// attach b to the parent of x
	if(x==root){
		root= b;
		arr[b].parent= INT_MIN;
	}else{
		int parent_x= arr[x].parent;
		arr[b].parent=parent_x;
		if(arr[parent_x].left == x)
			arr[parent_x].left = b;
		else arr[parent_x].right = b;
	}
	/* make    b
			  / \
			 a   c */
	arr[b].left= a;
	arr[a].parent= b;
	arr[b].right = c;
	arr[c].parent =b;

	/*attach t0, t1, t2 and t3*/
	arr[a].left = t0;
	if(t0!=INT_MIN) arr[t0].parent = a;
	arr[a].right = t1;
	if(t1!=INT_MIN) arr[t1].parent = a;

	arr[c].left= t2;
	if(t2!=INT_MIN) arr[t2].parent = c;
	arr[c].right = t3;
	if(t3!=INT_MIN) arr[t3].parent = c;
	recompute_height(a);
	recompute_height(c);
	return b;
}


template<typename T> __host__ __device__ 
void Tree<T>::init_node(Node<T>& node,int curr_ind, int pred_, int succ_, int parent_){
	arr[curr_ind].parent= parent_;
	arr[curr_ind].height=0;
	arr[curr_ind].pred= pred_;
	arr[curr_ind].succ= succ_;
	arr[succ_].pred= curr_ind;
	arr[pred_].succ= curr_ind;

	if(arr[parent_] < node)
		arr[parent_].right= curr_ind;
	else arr[parent_].left= curr_ind;
}

template<typename T> __host__ __device__ 
void Tree<T>::insert(T* key_){
	strcpy(arr[next_index].key,key_);
	int p= search(arr[next_index]);
	if(arr[p] == arr[next_index]){
		strcpy(arr[next_index].key,"");
		return;
	}
	int pred_= arr[p] < arr[next_index]?p: arr[p].pred;
	int succ_= arr[pred_].succ;
	init_node(arr[next_index],next_index,pred_,succ_,p);
	// // after insert maximum one node will get imbalanced
	recompute_height(p);
	int x,y,z;
	x=y=z= next_index;
	while(x!=root){
		
		if(abs(height(arr[x].left) - height(arr[x].right))<=1){
			z=y;
			y=x;
			x= arr[x].parent;
		}else break;
	}
	if(x!=root)
		tri_node_restructure(x,y,z);
	++next_index;
}

template<typename T> __host__ __device__ 
int Tree<T>::taller_child(int x){
	return (height(arr[x].left) > height(arr[x].right)?
						arr[x].left : arr[x].right);
}

template<typename T> __host__ __device__ 
void Tree<T>::delete_rebalance(int p){
	int x,y,z;
	while(p!=root){
		if(abs(height(arr[p].left)-height(arr[p].right))>1){
			x=p;
			y= taller_child(x);
			z= taller_child(y);
			p= tri_node_restructure(x,y,z);
		}
		p=arr[p].parent;
	}
}

template<typename T> __host__ __device__ 
void Tree<T>::delete_(T* key_){
	int p;
	int parent_;
	int succ_;

	p= search(key_);
	if(strcmp_(arr[p].key,key_)==0)
		return;
	// node has no children
	parent_ = arr[p].parent;
	if(arr[p].left==INT_MIN && arr[p].right == INT_MIN){
		if(arr[parent_].right == p)
			arr[parent_].right= INT_MIN;
		else arr[parent_].left = INT_MIN;
		recompute_height(parent_);
		delete_rebalance(parent_);
		return;
	}else if(arr[p].left==INT_MIN){  // when deleted node has only right child
		if(arr[parent_].left==p)
			arr[parent_].left= arr[p].right;
		else arr[parent_].right = arr[p].right;
		recompute_height(parent_);
		delete_rebalance(parent_);
		return;
	}else if(arr[p].right == INT_MIN){ // when deleted node has only left child
		if(arr[parent_].left==p)
			arr[parent_].left= arr[p].left;
		else arr[parent_].right= arr[p].left;
		recompute_height(parent_);
		delete_rebalance(parent_);
		return;
	}	// when deleted node has both children
	succ_ = arr[p].right;
	while(arr[succ_].left!=INT_MIN)
		succ_ = arr[succ_].left;
	strcpy(arr[p].key,arr[succ_].key);
	parent_= arr[succ_].parent;
	arr[parent_].left = arr[succ_].right;
	recompute_height(parent_);
	delete_rebalance(parent_);
	return;
}

template<typename T> __host__ __device__
bool Tree<T>::search_2(T* key_){
	int p=search(key_);
	Node<char> node= Node<char>(key_);
	while(arr[p]> node)
		p= arr[p].pred;
	int x= p;
	while(arr[p]< node)
		p= arr[p].succ;
	int y=p;
	
	return (arr[x] == node ||arr[y] == node);
}	

template<typename T> 
void Tree<T>::insert_nodes(char* filename,int max){
	FILE* fp= fopen(filename,"r");
	char buff[200];
	fgets(buff,200,fp);
	int size= atoi(buff);
	Tree<char>*tree= new Tree(size+max);
	while(fgets(buff,200,fp)!=NULL){
		buff[strlen(buff)-1]='\0';
		char* temp1= strtok(buff,"$");
		char* temp2= strtok(NULL,"$");
	// 	// Node<char> *node=new Node<char>(temp1);
		tree->insert(temp1);
	}
	fclose(fp);
	*this = *tree;
}
