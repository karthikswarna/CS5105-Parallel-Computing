#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <chrono>
#include <limits>
#include <mpi.h>

using namespace std;

#define GRAPH_SIZE 8   // Number of nodes in the graph.

int **alloc_2d_array(int, int);

int main(int argc, char **argv)
{
	int rank, np;
	fstream f;

	auto start = chrono::high_resolution_clock::now();
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD , &rank);
	MPI_Comm_size(MPI_COMM_WORLD , &np);

	if(rank == 0)
	{
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0, 2);

		// Grnerating a random graph
		int adj_matrix[GRAPH_SIZE][GRAPH_SIZE];
		for(int i = 0; i < GRAPH_SIZE; i++)
			for(int j = 0; j < GRAPH_SIZE; j++)
				adj_matrix[i][j] = dist(mt);

		// Saving the graph to a file.
		f.open("adj_matrix.txt");
		for(int i = 0; i < GRAPH_SIZE; i++)
		{
			for(int j = 0; j < GRAPH_SIZE; j++)
				f << adj_matrix[i][j] << ' ';
			f << '\n';
		}
		f.close();
	}

	MPI_Barrier(MPI_COMM_WORLD);

	int npr = sqrt(np);
	int block_size = (GRAPH_SIZE / npr);
	int start_row = floor(rank / npr) * block_size;
	int start_col = (rank % npr) * block_size;
	int **rows = alloc_2d_array(block_size, GRAPH_SIZE);
	int **cols = alloc_2d_array(GRAPH_SIZE, block_size);

	unsigned int current_row = -1;
	string line;
	f.open("adj_matrix.txt", ios::in);
	while(getline(f, line))
	{
		++current_row;
		if(current_row > GRAPH_SIZE)
			break;

		if (current_row >= start_row && current_row < start_row + block_size)
		{
			for(int i = 0; i < line.length(); i += 2)
    			rows[current_row % block_size][i/2] = line[i] - '0';
		}

		for(int i = 2*start_col; i < 2*(start_col + block_size); i += 2)
			cols[current_row][(i/2) % block_size] = line[i] - '0';
	}

	int **prod = alloc_2d_array(block_size, block_size);

	for(int i = 0; i < block_size; i++)
	{
		for(int j = 0; j < block_size; j++)
		{
			prod[i][j] = numeric_limits<int>::max(); // inf
			for(int k = 0; k < GRAPH_SIZE; k++)
			{
				prod[i][j] = min(prod[i][j], rows[i][k] + cols[k][j]);
			}
		}
	}

	// Calculating A^N.
	for(int x = 2; x < GRAPH_SIZE; x *= 2)
	{
		// Sending corresponding C block to neighbours.
		for(int i = 0; i < npr; i++)
		{
			if((i * npr) + (rank % npr) != rank)
				MPI_Send(prod, block_size*block_size, MPI_INT, (i * npr) + (rank % npr), 1, MPI_COMM_WORLD);
			if(i + floor(rank / npr) * npr != rank)
				MPI_Send(prod, block_size*block_size, MPI_INT, i + floor(rank / npr) * npr, 1, MPI_COMM_WORLD);
		}
		
		// Receiving the needed blocks from neighbours.********************* problem may be here ***********************
		int **temp = alloc_2d_array(block_size, block_size);
		for(int i = 0; i < npr; i++)
		{
			// Receive columns
			int recv_rank = (i * npr) + (rank % npr);
			if(recv_rank != rank)
			{
				MPI_Recv(&temp, block_size*block_size, MPI_INT, recv_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for(int j = floor(recv_rank / npr) * block_size; j < (floor(recv_rank / npr) * block_size) + block_size; j++)
				{
					for(int k = 0; k < block_size; k++)
						cols[j][k] = temp[j - (int)(floor(recv_rank / npr) * block_size)][k];
				}
			}

			recv_rank = i + floor(rank / npr) * npr;
			// Receive rows
			if(recv_rank != rank)
			{
				MPI_Recv(&temp, block_size*block_size, MPI_INT, recv_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for(int k = 0; k < block_size; k++)
				{
					for(int j = (recv_rank % npr) * block_size; j < ((recv_rank % npr) * block_size) + block_size; j++)
						rows[k][j] = temp[k][j - ((recv_rank % npr) * block_size)];
				}
			}
		}

		// Calculate pseudo-product.
		for(int i = 0; i < block_size; i++)
		{
			for(int j = 0; j < block_size; j++)
			{
				prod[i][j] = numeric_limits<int>::max(); // inf
				for(int k = 0; k < GRAPH_SIZE; k++)
				{
					prod[i][j] = min(prod[i][j], rows[i][k] + cols[k][j]);
				}
			}
		}
	}

	/************************************** Writing results into file *******************************/
	if(rank == 0)
	{
		f.open("product.txt");
		f << "Block C[0][0]: \n";
		for(int i = 0; i < block_size; i++)
		{
			for(int j = 0; j < block_size; j++)
				f << prod[i][j] << ' ';
			f << '\n';
		}
		f.close();

		MPI_Send(&rank, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
	}
	else if(rank == np - 1)
	{
		int temp_flag;
		MPI_Recv(&temp_flag, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		f.open("product.txt");
		f << "Block C[" << floor(rank / npr) << "][" << rank % npr << "]:\n";
		for(int i = 0; i < block_size; i++)
		{
			for(int j = 0; j < block_size; j++)
				f << prod[i][j] << ' ';
			f << '\n';
		}
		f.close();
	}
	else
	{
		int temp_flag;
		MPI_Recv(&temp_flag, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		f.open("product.txt");
		f << "Block C[" << floor(rank / npr) << "][" << rank % npr << "]:\n";
		for(int i = 0; i < block_size; i++)
		{
			for(int j = 0; j < block_size; j++)
				f << prod[i][j] << ' ';
			f << '\n';
		}
		f.close();

		MPI_Send(&rank, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
	}


	MPI_Finalize();
	auto finish = chrono::high_resolution_clock::now();

	auto time_spent = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
	cout << time_spent << " milli seconds" << endl;

	return 0;
}

int **alloc_2d_array(int rows, int cols)
{
    int *data = (int *)malloc(rows * cols * sizeof(int));
    int **array= (int **)malloc(rows * sizeof(int *));

    for (int i = 0; i < rows; i++)
        array[i] = &(data[cols * i]);

    return array;
}