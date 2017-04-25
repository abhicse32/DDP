#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
using namespace std;

/*structure of the nodes of the tree*/
template<class T>
class Node{
	public:
		int parent;
		T key;
		int height;
		int left;
		int right;
		int succ,pred;
		
		Node(T key_):key(key_){
			parent=left=right=INT_MIN;
			succ=pred=INT_MIN;
			height=INT_MIN;
		}

		Node(){
			new(this)Node(INT_MIN);
		}

		friend ostream &operator<<(ostream &os, Node& node){
			os << node.key;
			return os;
		}

		 bool operator<(Node &node){
			return key<node.key;
		}

		 bool operator==(Node& node){
			return key==node.key;
		}
		 bool operator>(Node &node){
			return key>node.key;
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

		Tree(int size_):root(1),next_index(2),size(size_){
			arr= new Node<T>[size];
			arr[root].key = INT_MAX;
			arr[root].parent = 0;
			arr[0].right= root;
			arr[root].pred= 0;
			arr[0].succ= root;
			arr[root].height=0;
		};

		Tree(){}
		 void inorder(int);
		 void preorder(int);
		 void postorder(int);
		 int search(Node<T>&);
		 int search(T&);	
         int height(int);
		 void insert(T&);
		 void delete_(T&);
		 void delete_rebalance(int);
		 void init_node(T&, int, int, int, int);
		 int tri_node_restructure(int, int, int);
		 void recompute_height(int);
		 int taller_child(int);
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

template<typename T>  
int Tree<T>::search(Node<T> &node){
	if(root==INT_MIN)
		return root;
    int temp=root;
    while(true){
        if(arr[temp]==node)
            return temp;
        int child= (arr[temp]<node? arr[temp].right: 
        	 			arr[temp].left);
        if(child==INT_MIN)
        	return temp;
        temp= child;
    }	
}

template<typename T> 
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
}

template<typename T>  
void Tree<T>::recompute_height(int x){
	while(x!=root){
		arr[x].height = max(height(arr[x].right), height(arr[x].left));
		x= arr[x].parent;
	}
}

template<typename T>   
int Tree<T>::height(int index){
	if(index==INT_MIN || arr[index].key==INT_MIN)
		return 0;
	return arr[index].height+1;
}
template<typename T> 
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


template<typename T> 
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

template<typename T> 
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

template<typename T> 
int Tree<T>::taller_child(int x){
	return (height(arr[x].left) > height(arr[x].right)?
						arr[x].left : arr[x].right);
}

template<typename T> 
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

template<typename T>
void Tree<T>::delete_(T& key_){
	int p;
	int parent_;
	int succ_;

	p= search(key_);
	if(arr[p].key != key_)
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
	arr[p].key= arr[succ_].key;
	parent_= arr[succ_].parent;
	arr[parent_].left = arr[succ_].right;
	recompute_height(parent_);
	delete_rebalance(parent_);
	return;
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