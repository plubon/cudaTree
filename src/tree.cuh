/*
 * tree.cuh
 *
 *  Created on: Sep 12, 2016
 *      Author: piotr
 */

#ifndef TREE_CUH_
#define TREE_CUH_
#define ORDER 512
#define INPUTSIZE 1000000
#define BLOCKSIZE 1024

typedef struct node {
	node ** pointers;
	int * keys;
	struct node * parent;
	bool is_leaf;
	int num_keys;
	struct node * next;
} node;

__device__ void make_node(node*&);
__device__ void make_leaf(node*&);
__global__ void bulkLoad(node*&, int*, int*);
__global__ void buildLeaves(node*&, int*, int*);
__global__ void fillLeaves(node*&, int*, int*);
__global__ void test(node*&, int*, int*);
__global__ void buildLevel(node*&, int*, int*, int, int);
__global__ void fillLevel(node*&, int*, int*, int, int);
__global__ void fillRoot(node*&, int*, int*, int, int);
__global__ void buildRoot(node*&, int*, int*);
__global__ void search(int, int*);
__global__ void search(int*, int*, int);
__global__ void insertVal(int);
__device__ node* find(int);
__device__ void addKey(node*, int);
__device__ int contains(node*, int);
__global__ void searchBetter(int val, int* result);
__global__ void containsBetter(int val, int* result);
__global__ void copyNode(int val, int* full);
__global__ void copyNode(node* node, int* full);
__global__ void addValue(int val);
__global__ void addValue(node* nn);
__global__ void createNewNode(node*& nnode);
__global__ void copyToNewNode(node*& nnode);
#endif /* TREE_CUH_ */
