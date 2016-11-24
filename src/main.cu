////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
// includes CUDA
#include <cuda_runtime.h>

// includes, project// helper functions for SDK examples
#include "tree.cuh"
#include <algorithm>
#include <fstream>
using namespace std;



bool findHost(int val)
{
	int* result = NULL;
	int host1[] = {2};
	cudaError_t err = cudaMalloc((void **)&result, 1 * sizeof(int));
	if (err != cudaSuccess)
		cout<<"result "<<cudaGetErrorString(err)<<endl;
	cudaMemcpy(result, host1, 1*sizeof(int), cudaMemcpyHostToDevice);
	cout<<host1[0]<<endl;
	while(host1[0] < 5)
	{
		searchBetter<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(val , result);
		err = cudaThreadSynchronize();
		cudaMemcpy(host1, result, 1*sizeof(int),  cudaMemcpyDeviceToHost);
		err = cudaThreadSynchronize();
		if(host1[0] > 0)
			cout<<host1[0]<<endl;
	}
	containsBetter<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(val, result);
	err = cudaThreadSynchronize();
	cudaMemcpy(host1, result, 1*sizeof(int),  cudaMemcpyDeviceToHost);
	err = cudaThreadSynchronize();
	if(host1[0] == 1)
	{
		cout<<"FOUND"<<endl;
		return true;
	}
	else
	{
		cout<<"NOT FOUND"<<endl;
		return false;
	}
}

void splitHost(node* node1)
{
	node* nn;
	createNewNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(nn);
	cudaThreadSynchronize();
	copyToNewNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(nn);
	cudaThreadSynchronize();
}

void insertHost(node* node)
{
	int* fullDevice = NULL;
	int fullHost[1];
	cudaError_t err = cudaMalloc((void **)&fullDevice, 1 * sizeof(int));
	cudaThreadSynchronize();
	copyNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(node, fullDevice);
	cudaMemcpy(fullHost, fullDevice, 1*sizeof(int),  cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	addValue<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(node);
	cudaThreadSynchronize();
	if(fullHost[0])
		splitHost(node);
}

void splitHost(int val)
{
	node* newNode;
	createNewNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(newNode);
	cudaThreadSynchronize();
	copyToNewNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(newNode);
	cudaThreadSynchronize();
	insertHost(newNode);
}

void insertHost(int val)
{
	if(findHost(val))
		return;
	int* fullDevice = NULL;
	int fullHost[1];
	cudaError_t err = cudaMalloc((void **)&fullDevice, 1 * sizeof(int));
	cudaThreadSynchronize();
	copyNode<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(val, fullDevice);
	cudaMemcpy(fullHost, fullDevice, 1*sizeof(int),  cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	addValue<<<(ORDER + BLOCKSIZE -1) / BLOCKSIZE, BLOCKSIZE>>>(val);
	cudaThreadSynchronize();
	if(fullHost[0])
		splitHost(val);
}

int main(int argc, char **argv)
{
	string filename="/home/piotr/Uczelnia/Cuda/BTree/testfile";
	int* initValues = (int*)malloc(INPUTSIZE*sizeof(int));
	node* root;
	ifstream file(filename.c_str());
	string line;
	int i =0;
	while(getline(file, line))
	{
		initValues[i] = atoi(line.c_str());
		i++;
	}
	sort(initValues, initValues+INPUTSIZE);
    int* input = NULL;
    int* result = NULL;
    int* results = NULL;
    int* values = NULL;
    int myArray[3] = { -1, 898386, 40156 };
    int host[1], hostResults[3];
    cudaError_t err = cudaMalloc((void **)&input, INPUTSIZE * sizeof(int));
    if (err != cudaSuccess)
    	cout<<"input "<<cudaGetErrorString(err)<<endl;
    err = cudaMalloc((void **)&root, sizeof(node));
	if (err != cudaSuccess)
		cout<<"root "<<cudaGetErrorString(err)<<endl;
    err = cudaMalloc((void **)&result, 1 * sizeof(int));
    if (err != cudaSuccess)
    	cout<<"result "<<cudaGetErrorString(err)<<endl;
    err = cudaMalloc((void **)&results, 3 * sizeof(int));
	if (err != cudaSuccess)
		cout<<"results "<<cudaGetErrorString(err)<<endl;
	err = cudaMalloc((void **)&values, 3 * sizeof(int));
	if (err != cudaSuccess)
		cout<<"values "<<cudaGetErrorString(err)<<endl;
    err = cudaMemcpy(input, initValues, INPUTSIZE*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		cout<<"copy input "<<cudaGetErrorString(err)<<endl;
	err = cudaMemcpy(values, myArray, 3*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
			cout<<"copy values "<<cudaGetErrorString(err)<<endl;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512000000);
    buildLeaves<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
    	cout<<"buildLeaves "<<cudaGetErrorString(err)<<endl;
    fillLeaves<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
    	cout<<"fillLeaves "<<cudaGetErrorString(err)<<endl;
    int size = INPUTSIZE / ((ORDER/2)-1);
    int sw = 1;
    int level =0;
    while(size > ORDER)
    {
    	cout<<"Size of level "<<level<<": "<<size<<endl;
    	buildLevel<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result, size, sw);
    	err = cudaThreadSynchronize();
		if (err != cudaSuccess)
			cout<<"buildlevel "<<cudaGetErrorString(err)<<endl;
    	fillLevel<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result, size, sw);
    	err = cudaThreadSynchronize();
		if (err != cudaSuccess)
			cout<<"filllevel "<<cudaGetErrorString(err)<<endl;
    	size = size /(ORDER/2);
    	sw = 1 - sw;
    	level++;
    }
    cout<<"Size of level "<<level<<": "<<size<<endl;
    buildRoot<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess)
		cout<<"build root "<<cudaGetErrorString(err)<<endl;
	fillRoot<<<INPUTSIZE/BLOCKSIZE + 1, BLOCKSIZE>>>(root, input, result, size, sw);
	if (err != cudaSuccess)
		cout<<"fill root "<<cudaGetErrorString(err)<<endl;
	err = cudaThreadSynchronize();
    //test<<<(INPUTSIZE/BLOCKSIZE), BLOCKSIZE>>>(root, input, result);
	int v = -1;
	findHost(97861);
	insertHost(-1);
	findHost(-1);
	search<<<1, BLOCKSIZE>>>(v, result);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
       	cout<<"1st search call "<<cudaGetErrorString(err)<<endl;
    cudaMemcpy(host, result, 1*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        	cout<<"copying result "<<cudaGetErrorString(err)<<endl;

    int f = 0;
    for(int i=0; i<INPUTSIZE; i++)
    {
    	if(initValues[i] == v)
    		f = 1;
    }
    cout<<host[0]<<" "<<f<<endl;
    search<<<3, BLOCKSIZE>>>(values, results, 3);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
    		cout<<"multiple search call "<<cudaGetErrorString(err)<<endl;
    cudaMemcpy(hostResults, results, 3*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<3; i++)
    	cout<<hostResults[i]<<endl;
    insertVal<<<1, BLOCKSIZE>>>(v);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
    	cout<<"insert call "<<cudaGetErrorString(err)<<endl;
    search<<<1, BLOCKSIZE>>>(v, result);
    err = cudaThreadSynchronize();
    if (err != cudaSuccess)
		cout<<"2nd search call "<<cudaGetErrorString(err)<<endl;
    cudaMemcpy(host, result, 1*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    	cout<<"copying result "<<cudaGetErrorString(err)<<endl;
    cout<<host[0]<<endl;
    cout<<"end"<<endl;
    cudaFree(input);
    cudaFree(result);
}
