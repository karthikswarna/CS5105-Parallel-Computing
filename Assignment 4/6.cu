%%cu
/*
 * INPUT:
 * The number of nodes and source vertex is given as input.
 * 0 <= SOURCE < NUM_NODES
 * OUTPUT:
 * If a node is not reachable from the source, 2147483647(Infinity) will be the shortest distance from s to d.
 */

#include <iostream>
#include <stdlib.h>
#include <time.h>

#define NUM_NODES 5
#define SOURCE 0

using namespace std;

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
	{
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void CUDA_SSSP_KERNEL1(int *Va, int *Ea, int *Wa, bool *Ma, int *Ca, int *Ua, bool *done)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid > NUM_NODES)
		*done = false;

	if (Ma[tid] == true)
	{
		Ma[tid] = false;
		
		__syncthreads();
		
		int start = Va[tid];
		int end = Va[tid + 1] - 1;
		for (int i = start; i <= end; i++) 
		{
			int nid = Ea[i];

			if(Ua[nid] > Ca[tid] + Wa[i])
			{
                Ua[nid] = Ca[tid] + Wa[i];
				*done = false;
			}
		}
	}
}

__global__ void CUDA_SSSP_KERNEL2(int *Va, int *Ea, int *Wa, bool *Ma, int *Ca, int *Ua, bool *done)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(Ca[tid] > Ua[tid])
    {
        Ca[tid] = Ua[tid];
        Ma[tid] = true;
    }
    Ua[tid] = Ca[tid];
}

int main()
{
	srand(time(NULL));

	/* Generating a random graph */
	cout << "Adjacent Matrix:" << endl;

	int adj_matrix[NUM_NODES][NUM_NODES];
	int NUM_EDGES = 0;
	for(int i = 0; i < NUM_NODES; i++)
	{
		for(int j = 0; j < NUM_NODES; j++)
		{
			adj_matrix[i][j] = rand() % 100;	// Weights can be from 0 to 100.
			if(adj_matrix[i][j] != 0)
				NUM_EDGES++;

			cout << adj_matrix[i][j] << " ";
		}
		cout << endl;
	}

	/* Creating vertex and edges array using the adjacency matrix in CPU */
	int vertices[NUM_NODES];
	int *edges = new int[NUM_EDGES];
	int *weights = new int[NUM_EDGES];
	int index = 0;
	vertices[0] = 0;
	for(int i = 0; i < NUM_NODES; i++)
	{
		// Finding the number of edges for ith vertex.
		for(int j = 0; j < NUM_NODES; j++)
		{
			if(adj_matrix[i][j] != 0)
			{
				edges[index] = j;
				weights[index] = adj_matrix[i][j];
				index++;
			}
		}

		vertices[i + 1] = index;
		if(vertices[i] == vertices[i + 1])
			vertices[i] = -1;
	}

	cout << "Vertices Array:" << endl;
	for(int i = 0; i < NUM_NODES; i++)
		cout << vertices[i] << " ";

	cout << endl << "Edges Array:" << endl;
	for(int i = 0; i < NUM_EDGES; i++)
		cout << edges[i] << " ";

	cout << endl << "Weights Array:" << endl;
	for(int i = 0; i < NUM_EDGES; i++)
		cout << weights[i] << " ";
	cout << endl;

	/* Creating and Initializing the Mask, Cost, UpdateCost arrays in CPU. */
	bool mask[NUM_NODES] = {false};
	int cost[NUM_NODES];
    int updateCost[NUM_NODES];
    for (int i = 0; i < NUM_NODES ; i++)
    {
        cost[i] = INT_MAX;
        updateCost[i] = INT_MAX;
    }

	mask[SOURCE] = true;
	cost[SOURCE] = 0;
    updateCost[SOURCE] = 0;


	/* Allocating the memory and initializing the Vertex, Edges, Frontier, Visited, Cost arrays in GPU */
	int* Va, *Ea, *Wa, *Ca, *Ua;
    bool* Ma;

	HANDLE_ERROR( cudaMalloc((void**)&Va, sizeof(int)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ea, sizeof(int)*NUM_EDGES) );
	HANDLE_ERROR( cudaMalloc((void**)&Wa, sizeof(int)*NUM_EDGES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ma, sizeof(bool)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ua, sizeof(int)*NUM_NODES) );

	HANDLE_ERROR( cudaMemcpy(Va, vertices, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ea, edges, sizeof(int)*NUM_EDGES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Wa, weights, sizeof(int)*NUM_EDGES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ma, mask, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ua, updateCost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice) );

	// int threadsPerBlock = 512;
    // int blocksPerGrid = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;

	int blocksPerGrid = 1;
	int threadsPerBlock = NUM_NODES;

	bool done;
	bool* d_done;
	HANDLE_ERROR( cudaMalloc((void**)&d_done, sizeof(bool)) );

  	done = false;
	while (!done)
    {
		done = true;
		HANDLE_ERROR( cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice) );
		CUDA_SSSP_KERNEL1 <<<blocksPerGrid, threadsPerBlock>>>(Va, Ea, Wa, Ma, Ca, Ua, d_done);
		HANDLE_ERROR( cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost) );

		CUDA_SSSP_KERNEL2 <<<blocksPerGrid, threadsPerBlock>>>(Va, Ea, Wa, Ma, Ca, Ua, d_done);
	}

	HANDLE_ERROR( cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost) );

	cout << endl << "Shortest Path Costs from vertex " << SOURCE << ":  (2147483647 means infinity)" << endl;
	for (int i = 0; i < NUM_NODES; i++)
		cout << cost[i] << " ";
	cout << endl;

	delete[] edges;
	delete[] weights;
    cudaFree (Va);
    cudaFree (Ea);
    cudaFree (Wa);
    cudaFree (Ma);
    cudaFree (Ca);
    cudaFree (Ua);
    cudaFree (d_done);

    return 0;
}