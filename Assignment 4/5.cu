%%cu
/*
 * INPUT:
 * The number of nodes and source vertex is given as input.
 * 0 <= SOURCE < NUM_NODES
 * OUTPUT:
 * If a node is not reachable from the source, 2147483647(Infinity) will be the number of edges in the path from s to d.
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

__global__ void CUDA_BFS_KERNEL(int *Va, int *Ea, bool *Fa, bool *Xa, int *Ca, bool *done)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > NUM_NODES)
	{
		*done = false;
		return;
	}
	if (Fa[id] == true && Xa[id] == false)
	{
		Fa[id] = false;
		Xa[id] = true;
		
		__syncthreads();
		
		int start = Va[id];
		int end = Va[id + 1] - 1;
		for (int i = start; i <= end; i++) 
		{
			int nid = Ea[i];		// Neighbour thread id.

			if (Xa[nid] == false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}
		}
	}
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
			adj_matrix[i][j] = rand() % 2;
			if(adj_matrix[i][j] == 1)
				NUM_EDGES++;

			cout << adj_matrix[i][j] << " ";
		}
		cout << endl;
	}

	/* Creating vertex and edges array using the adjacency matrix in CPU */
	int vertices[NUM_NODES];
	int *edges = new int[NUM_EDGES];
	int index = 0;
	vertices[0] = 0;
	for(int i = 0; i < NUM_NODES; i++)
	{
		// Finding the number of edges for ith vertex.
		for(int j = 0; j < NUM_NODES; j++)
		{
			if(adj_matrix[i][j] == 1)
				edges[index++] = j;
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
	cout << endl;


	/* Creating and Initializing the Frontier, Visited, Cost arrays in CPU. */
	bool frontier[NUM_NODES] = {false};
	bool visited[NUM_NODES] = {false};
	int cost[NUM_NODES];
    for (int i = 0; i < NUM_NODES ; i++)
        cost[i] = INT_MAX;
	
	frontier[SOURCE] = true;
	cost[SOURCE] = 0;


	/* Allocating the memory and initializing the Vertex, Edges, Frontier, Visited, Cost arrays in GPU */
	int *Va, *Ea, *Ca;
	bool *Fa, *Xa;

	HANDLE_ERROR( cudaMalloc((void**)&Va, sizeof(int)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ea, sizeof(int)*NUM_EDGES) );
	HANDLE_ERROR( cudaMalloc((void**)&Fa, sizeof(bool)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Xa, sizeof(bool)*NUM_NODES) );
	HANDLE_ERROR( cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES) );

	HANDLE_ERROR( cudaMemcpy(Va, vertices, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ea, edges, sizeof(int)*NUM_EDGES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Fa, frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Xa, visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice) );

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
		CUDA_BFS_KERNEL <<<blocksPerGrid, threadsPerBlock>>>(Va, Ea, Fa, Xa, Ca, d_done);
		HANDLE_ERROR( cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost) );
	}

	HANDLE_ERROR( cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost) );

	cout << endl << "Minimum path costs from vertex " << SOURCE << ":  (2147483647 means infinity)" << endl;
	for (int i = 0; i < NUM_NODES; i++)
		cout << cost[i] << " ";
	cout << endl;

	delete[] edges;
	cudaFree (Va);
	cudaFree (Ea);
	cudaFree (Fa);
	cudaFree (Xa);
	cudaFree (Ca);
	cudaFree (d_done);

	return 0;
}