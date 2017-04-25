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
using namespace std;

template<class T>
class Node{
	public:
		int parent;
		T key;
		int height;
		int left;
		int right;
		int succ,pred;
		bool mark;
		
		__host__ __device__ Node(T key_):key(key_){
			parent=left=right=INT_MIN;
			succ=pred=INT_MIN;
			height=INT_MIN;
			mark= false;
		}

		__host__ __device__ Node(){
			new(this)Node(INT_MIN);
		}

		friend ostream &operator<<(ostream &os, Node& node){
			os << node.key;
			return os;
		}

		__host__ __device__ bool operator<(Node &node){
			return key<node.key;
		}

		__host__ __device__ bool operator==(Node& node){
			return key==node.key;
		}
		__host__ __device__ bool operator>(Node &node){
			return key>node.key;
		}
};

/*Definition of the tree*/
template<class T>
class Tree{
	public:
		int root,next_index;
		int size;

		// these three fields useful only when concurrent inserts
		// and deletes work
	    int* succ_locks, * tree_locks;
		int *tree_lock_thread;


		/*container of the tree*/
		//node at index 0 is sentinel node with key INT_MIN
		// initialized root with key INT_MAX and index=1
		Node<T>* arr;

		__host__ __device__ Tree(int size_):root(1),next_index(2),size(size_){
			arr= new Node<T>[size];
			arr[root].key = INT_MAX;
			arr[root].parent = root;
			arr[0].right= root;
			arr[root].pred= 0;
			arr[0].succ= root;
			arr[root].height=0;
		};

		__host__ __device__ Tree(){}
		void inorder(int);
		void preorder(int);
		void postorder(int);
		__host__ __device__ int search(Node<T>&);
		__host__ __device__ int search(T&);	
        __host__ __device__ int height(int);
		__host__ __device__ void insert(T&);
		__host__ __device__ void delete_(T&);
		__host__ __device__ void delete_rebalance(int);
		__host__ __device__ void init_node(T&, int, int, int, int);
		__host__ __device__ int tri_node_restructure(int, int, int);
		__host__ __device__ void recompute_height(int);
		__host__ __device__ int taller_child(int);
		__host__ __device__ bool search_2(T&);

		__device__ bool d_insert(T&, int);
		__device__ int choose_parent(int, int, int, int);
		__device__ int lock_parent(int,int);
		__device__ void rebalancing(int, int, int,bool);
	    __device__ int restart(int, int,int);
	    __device__ void rotate(int,int,int,bool);
	    __device__ int get_balance_factor(int);
	    __device__ bool remove(T&, int);
	    __device__ int acquire_tree_locks(int, int);
	    __device__ void remove_from_tree(int,int,int,int);
	    __device__ bool update_child(int, int, int);
		void insert_nodes(char*,int);

};

template<class T> 
void Tree<T>::inorder(int index){
	if(index==INT_MIN || arr[index].key==INT_MIN)
		return;
	inorder(arr[index].left);
        cout<< arr[index] <<" ";
	inorder(arr[index].right);
}

template<class T>
void Tree<T>::preorder(int index){
	if(index==INT_MIN || arr[index].key==INT_MIN)
		return;
	cout << arr[index] <<" ";
	preorder(arr[index].left);
	preorder(arr[index].right);
}

template<class T> 
void Tree<T>::postorder(int index){
	if(index==INT_MIN || arr[index].key==INT_MIN)
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
}

template<typename T> __host__ __device__
int Tree<T>::search(T &key_){
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
	return INT_MIN;
}

template<typename T> __host__ __device__
void Tree<T>::recompute_height(int x){
	while(x!=root){
		arr[x].height = max(height(arr[x].right), height(arr[x].left));
		x= arr[x].parent;
	}
}

template<typename T> __host__ __device__  
int Tree<T>::height(int index){
	if(index==INT_MIN || arr[index].key==INT_MIN)
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
	int a,b,c;
	int t0,t1,t2,t3;
	if(z_is_left_child && y_is_left_child){
		a= z; b = y; c= x;
		t0 = arr[z].left; t1= arr[z].right;
		t2= arr[y].right; t3= arr[x].right; 
	}else if(!z_is_left_child && y_is_left_child){
		a= y; b=z; c= x;
		t0= arr[y].left; t1= arr[z].left;
		t2= arr[z].right; t3= arr[x].right;
	}else if(z_is_left_child && !y_is_left_child){
		a= x; c= y; b=z;
		t0= arr[x].left; t1= arr[z].left;
		t2= arr[z].right; t3= arr[y].right;
	}else{
		a=x; b=y; c=z;
		t0= arr[x].left; t1= arr[y].left;
		t2= arr[z].left; t3= arr[z].right;
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
void Tree<T>::init_node(T& key_,int curr_ind, int pred_, int succ_, int parent_){
	arr[curr_ind].key=key_;
	arr[curr_ind].parent= parent_;
	arr[curr_ind].height=0;
	arr[curr_ind].pred= pred_;
	arr[curr_ind].succ= succ_;
	arr[succ_].pred= curr_ind;
	arr[pred_].succ= curr_ind;
	if(arr[parent_].key < key_)
		arr[parent_].right= curr_ind;
	else arr[parent_].left= curr_ind;
}

template<typename T> __host__ __device__
void Tree<T>::insert(T &key_){
	int p= search(key_);
	if(arr[p].key== key_)
		return;
	int pred_= arr[p].key < key_?p: arr[p].pred;
	int succ_= arr[pred_].succ;

	init_node(key_,next_index,pred_,succ_,p);
	// after insert maximum one node will get imbalanced
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
void Tree<T>::delete_(T& key_){
	int p;
	int parent_;
	int succ_;

	p= search(key_);
	if(arr[p].key != key_)
		return;
	// node has no children
	arr[arr[p].succ].pred= arr[p].pred;
	arr[arr[p].pred].succ= arr[p].succ;

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
	arr[p].key= arr[succ_].key;
	parent_= arr[succ_].parent;
	arr[parent_].left = arr[succ_].right;
	recompute_height(parent_);
	delete_rebalance(parent_);
	return;
}

template<typename T> __host__ __device__
bool Tree<T>::search_2(T& key_){
	int p=search(key_);
	while(arr[p].key > key_)
		p= arr[p].pred;
	int x= p;
	while(arr[p].key < key_)
		p= arr[p].succ;
	int y=p;
	return (arr[x].key == key_ || arr[y].key == key_);
}	

template<typename T> 
void Tree<T>::insert_nodes(char* filename, int max){
	FILE* fp= fopen(filename,"r");
	int node_;
	fscanf(fp,"%d",&size);
	Tree<int>*tree= new Tree<int>(size+max);
	while(fscanf(fp,"%d",&node_)!=EOF)
		tree->insert(node_);

	fclose(fp);
	*this = *tree;
}

// device code implemented but does not work 
// hence came up with another idea which is implemented
// above
template<typename T> __device__
bool Tree<T>::d_insert(T& key_, int thread_id){
	while(true){
		int node= search(key_);
		//printf("came here\n");
		int pred_= (arr[node].key < key_?node:arr[node].pred);
		bool flag=false;
		while(!flag){
			if(atomicExch(&(succ_locks[pred_]),1)==0){
				// critical section for this type of lock
				int succ_= arr[pred_].succ;
				if(key_ >arr[pred_].key && key_<= arr[succ_].key && !arr[pred_].mark){
					if(arr[succ_].key >= key_){
						atomicExch(&(succ_locks[pred_]),0);
						return false;
					}
					printf("%d\n",key_);
					int parent_= choose_parent(node,pred_,succ_,thread_id);
					init_node(key_,next_index,pred_,succ_,parent_);
					arr[parent_].height= max(arr[parent_].right, arr[parent_].left);
					++next_index;
					atomicExch(&(succ_locks[pred_]),0);
					if(parent_!=root){
						int grand_parent= lock_parent(parent_,thread_id);
						rebalancing(grand_parent,parent_,thread_id, arr[grand_parent].left==parent_);
					}else{
						// release the lock and remove the thread number from temp array
						tree_lock_thread[parent_]=-1;
						atomicExch(&(tree_locks[parent_]),0);
					}
					return true;
				}
				flag= true;
				atomicExch(&(succ_locks[pred_]),0);
			}
		}
		atomicExch(&(succ_locks[pred_]),0);
	}
}	


template<typename T> __device__
int Tree<T>::choose_parent(int first_cand, int pred_, int succ_, int thread_id){
	int candidate= (first_cand==pred_ || first_cand==succ_ ? first_cand:pred_);  
	// arbitrarily choose pred_ as the parent
	while(true){
		bool flag=false;
		while(!flag){
			if(atomicExch(&(tree_locks[candidate]),1)==0){
				tree_lock_thread[candidate]= thread_id;
				if(candidate==pred_){
					if(arr[candidate].right==INT_MIN)
						return candidate;
					tree_lock_thread[candidate]=-1;
					flag=true;
					atomicExch(&(tree_locks[candidate]),0);
					candidate= succ_;
				}else{
					if(arr[candidate].left==INT_MIN)
						return candidate;
					tree_lock_thread[candidate]=-1;
					flag=true;
					atomicExch(&(tree_locks[candidate]),0);
					candidate= pred_;
				}
			}
		}	
	}
}

template<typename T> __device__
int Tree<T>::lock_parent(int node, int thread_id){
	while(true){
		int parent_= arr[node].parent;
		bool flag=false;
		while(!flag){
			if(atomicExch(&(tree_locks[parent_]),1)==0){
				tree_lock_thread[parent_]=thread_id;
				if(arr[node].parent==parent_ && !arr[parent_].mark)
					return parent_;
				tree_lock_thread[parent_]=-1;
				flag=true;
				atomicExch(&(tree_locks[parent_]),0);
			}
		}
	}
}

template<typename T> __device__
int Tree<T>::restart(int node, int parent_,int thread_id){
	if(parent_!=INT_MIN){
		tree_lock_thread[parent_]=-1;
		atomicExch(&(tree_locks[parent_]),0);
	}
	tree_lock_thread[node]=-1;
	atomicExch(&tree_locks[node],0);
	while(true){
		bool flag=false;
		while(!flag){
			if(atomicExch(&(tree_locks[node]),1)==0){
				tree_lock_thread[node]=thread_id;
				if(arr[node].mark){
					tree_lock_thread[node]=-1;
					atomicExch(&(tree_locks[node]),0);
					return INT_MIN;
				}
				int child= get_balance_factor(node)>=2?arr[node].left:arr[node].right;
				if(child==INT_MIN)
					return INT_MIN;
				if(atomicCAS(&(tree_locks[child]),0,1)==0){
					tree_lock_thread[child]= thread_id;
					return child;
				}
				tree_lock_thread[node]=-1;
				flag=true;
				atomicExch(&(tree_locks[node]),0);
			}
		}
	}
}

template<typename T> __device__
void Tree<T>::rebalancing(int node, int child, int thread_id, bool is_left){
	// using relaxed balance
	if(node==root){
		tree_lock_thread[node]=-1;
		atomicExch(&(tree_locks[node]),0);
		if(child!=INT_MIN){
			tree_lock_thread[child]=-1;
			atomicExch(&(tree_locks[child]),0);
		}
		return;
	}

	int parent_=INT_MIN;
	
	while(node!=root){
		int new_height= max(height(arr[node].left),
						height(arr[node].right));

		int bf= get_balance_factor(node);
		if(new_height == arr[node].height && abs(bf) < 2)
			return;
		while(abs(bf)>=2){
			if((is_left && bf <= -2) ||(!is_left && bf >=2)){
				if(child!=INT_MIN){
					tree_lock_thread[child]=-1;
					atomicExch(&(tree_locks[child]),0);
				}
				child= is_left ? arr[node].right : arr[node].left;
				if(atomicCAS(&(tree_locks[child]),0,1)==1){
					child=restart(node,parent_,thread_id);
					if(tree_locks[node]!=thread_id)
						return;

					parent_= INT_MIN;
					is_left= (arr[node].left==child);
					bf = get_balance_factor(node);
					continue;
				}
				is_left = !is_left;
			}
			int child_bf= get_balance_factor(child);
			if((is_left && child_bf < 0) || (!is_left && child_bf > 0)){
				int grand_child= is_left?arr[child].right: arr[child].left;
				if(atomicCAS(&(tree_locks[grand_child]),0,1)==1){
					tree_lock_thread[child]=-1;
					atomicExch(&(tree_locks[child]),0);
					child= restart(node,parent_,thread_id);
					if(tree_lock_thread[node]!=thread_id)
						return;
					parent_= INT_MIN;
					is_left= (arr[node].left ==child);
					bf= get_balance_factor(node);
					continue;
				}
				rotate(grand_child,child,node,is_left);
				tree_lock_thread[child]=-1;
				atomicExch(&(tree_locks[child]),0);
				child= grand_child;
			}
			if(parent_==INT_MIN)
				parent_= lock_parent(node,thread_id);
			rotate(child,node,parent_,!is_left);
			bf= get_balance_factor(node);
			if(bf>=2 || bf <= -2){
				tree_lock_thread[parent_]=-1;
				atomicExch(&(tree_locks[parent_]),0);
				parent_= child;
				child= INT_MIN;
				is_left= bf>=2?false:true;
				continue;
			}
			int temp= child;
			child= node;
			node= temp;
			is_left=(arr[node].left == child);
			bf= get_balance_factor(node);

			if(child !=INT_MIN){
				tree_lock_thread[child]=-1;
				atomicExch(&(tree_locks[child]),0);
			}

			child= node;
			node= parent_!=INT_MIN && tree_lock_thread[parent_]==thread_id?
												parent_:lock_parent(node,thread_id); 
			is_left= (arr[node].left==child);
			parent_= INT_MIN;
		}
	}
	return;
}

template<typename T> __device__
int Tree<T>::get_balance_factor(int index){
	int left_h= height(arr[index].left);
	int right_h= height(arr[index].right);
		arr[index].height= max(left_h,right_h);
	return (left_h - right_h);
}

template<typename T> __device__
void Tree<T>::rotate(int child, int node, int parent_, bool left){
	if(left)
		arr[parent_].left= child;
	else arr[parent_].right= child;

	arr[child].parent=parent_;
	arr[node].parent= child;

	int grand_child= left?arr[child].left : arr[child].right;
	if(left){
		arr[node].right = grand_child;
		if(grand_child!=INT_MIN)
			arr[grand_child].parent= node;
		arr[child].left= node;
	}else{
		arr[node].left= grand_child;
		if(grand_child!=INT_MIN)
			arr[grand_child].parent=node;
		arr[child].right= node;
	}
	arr[node].height= max(height(arr[node].right),height(arr[node].left));
	arr[child].height= max(height(arr[child].left),height(arr[child].right));
}

template<typename T> __device__
bool Tree<T>::remove(T &key_, int thread_id){
	int node= search(key_);
	int pred_= arr[node].key > key_?arr[node].pred:node;
	bool flag= false;
	while(!flag){
		if(atomicExch(&(succ_locks[pred_]),1)==0){
			int succ_= arr[pred_].succ;
			if(key_ > arr[pred_].key && key_<=arr[succ_].key && !arr[pred_].mark){
				if(arr[succ_].key >key_){
					atomicExch(&(succ_locks[succ_]),0);
					return false;
				}
				bool succ_flag= false;
				while(!succ_flag){
					if(atomicExch(&(succ_locks[succ_]),1)==0){
						int succ_2= acquire_tree_locks(succ_,thread_id);
						int succ_parent= lock_parent(succ_,thread_id);
						arr[succ_].mark=true;
						int succ_succ= arr[succ_].succ;
						arr[succ_succ].pred= pred_;
						arr[pred_].succ= succ_succ;
						flag= true;
						succ_flag= true;
						atomicExch(&succ_locks[pred_],0);
						atomicExch(&(succ_locks[succ_]),0);
						remove_from_tree(succ_,succ_2,succ_parent,thread_id);
						return false;
					}
				}
			}
			atomicExch(&(succ_locks[pred_]),0);
			flag=true;
		}
	}
	return true;
}

template<typename T> __device__
int Tree<T>::acquire_tree_locks(int node, int thread_id){
	while(true){
		bool node_flag= false;
		while(!node_flag){
			if(atomicExch(&(tree_locks[node]),1)==0){
				tree_lock_thread[node]= thread_id;
				int right_= arr[node].right;
				int left_= arr[node].left;
				if(right_ == INT_MIN || left_ == INT_MIN){
					if(right_ != INT_MIN && atomicCAS(&(tree_locks[right_]),0,1)==1){
						tree_lock_thread[node]=-1;
						node_flag= true;
						atomicExch(&(tree_locks[node]),0);
						continue;
					}
					if(left_ == INT_MIN && atomicCAS(&tree_locks[left_],0,1)==1){
						tree_lock_thread[node]=-1;
						node_flag= true;
						atomicExch(&tree_locks[node],0);
						continue;
					}
					return INT_MIN;
				}
				// if node has two children, lock its successor(s)
				// s's parent and it's child if it exists
				int succ_= arr[node].succ;
				int parent_= arr[succ_].parent;
				if(parent_!=node){
					if(atomicCAS(&tree_locks[parent_],0,1)==1){
						tree_lock_thread[node]= -1;
						node_flag= true;
						atomicExch(&tree_locks[node],0);
						continue;
					}
				}else if(parent_!= arr[succ_].parent && arr[parent_].mark){
					tree_lock_thread[parent_]=-1;
					tree_lock_thread[node]= -1;
					atomicExch(&tree_locks[parent_],0);
					node_flag= true;
					atomicExch(&tree_locks[node],0);
					continue;
				}

				if(atomicCAS(&tree_locks[succ_],0,1)==1){
					tree_lock_thread[node]= -1;
					atomicExch(&tree_locks[node],0);
					if(parent_!=node){
						tree_lock_thread[parent_]=-1;
						atomicExch(&tree_locks[parent_],0);
					}
					continue;
				}

				int succ_right= arr[succ_].right;
				if(succ_right!=INT_MIN && atomicCAS(&tree_locks[succ_right],0,1)==1){
					tree_lock_thread[node]= -1;
					node_flag=true;
					atomicExch(&tree_locks[node],0);
					tree_lock_thread[succ_]= -1;
					atomicExch(&tree_locks[succ_],0);
					if(parent_!= node){
						tree_lock_thread[parent_]=-1;
						atomicExch(&tree_locks[parent_],0);
					}
					continue;
				}
				return succ_;
			}
		}
	}
}

template<typename T> __device__
void Tree<T>::remove_from_tree(int node, int succ_, int parent_, int thread_id){
	if(succ_==INT_MIN){
		int right_=arr[node].right;
		int child= (arr[node].right==INT_MIN ? arr[node].left : right_);
		int left= update_child(parent_,node,child);

		tree_lock_thread[node]=-1;
		atomicExch(&tree_locks[node],0);
		rebalancing(parent_,child,thread_id,left);
		return;
	}
 	int old_parent=arr[succ_].parent;
 	int old_right= arr[succ_].right;
 	update_child(old_parent,succ_,old_right);
 	arr[succ_].height= max(height(arr[node].left), height(arr[node].right));
 	
 	int left_ = arr[node].left;
 	int right_ = arr[node].right;
 	arr[succ_].parent= parent_;
 	arr[succ_].left= left_;
 	arr[succ_].right = right_;
 	arr[left_].parent = succ_;

 	if(right_!=INT_MIN)
 		arr[right_].parent= succ_;
 	if(arr[parent_].left==node)
 		arr[parent_].left=succ_;
 	else arr[parent_].right= succ_;

 	bool is_left=(old_parent!=node);
 	bool violated= abs(get_balance_factor(succ_))>=2;
 	if(is_left)
 		old_parent=succ_;
 	else{
 		tree_lock_thread[succ_]=-1;
 		atomicExch(&tree_locks[succ_],0);
 	}
 	tree_lock_thread[node]=-1;
 	atomicExch(&tree_locks[node],0);
 	tree_lock_thread[parent_]=-1;
 	atomicExch(&tree_locks[parent_],0);
 	rebalancing(old_parent,old_right,thread_id,is_left);
 	
 	if(violated){
 		bool succ_flag= false;
 		while(!succ_flag){
 			if(atomicExch(&tree_locks[succ_],1)==0){
 				int bf= get_balance_factor(succ_);
 				if(!arr[succ_].mark && abs(bf)>=2){
 					rebalancing(succ_,INT_MIN,thread_id, (bf>=2?false:true));
 				}else{
 					tree_lock_thread[succ_]=-1;
 					atomicExch(&tree_locks[succ_],0);
 				}
 			}
 		}
 	}
}

template<typename T> __device__
bool Tree<T>::update_child(int parent_, int old_child, int new_child){
	if(new_child!=INT_MIN)
		arr[new_child].parent= parent_;
	bool left= arr[parent_].left==old_child;
	if(left)
		arr[parent_].left= new_child;
	else arr[parent_].right= new_child;
	return left;
}
