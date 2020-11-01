/*
 * Author: S. Karthik Chandra 
 * Roll  : CS17B026
 */

#include <iostream>
#include <chrono>
#include <random>
#include "mpi.h"

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "Size of the array is expected as argument!\nFormat: mpiexec -np <proc> ./a.out <size>" << endl;
        exit(0);
    }

    int np, rank, length;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD , &np);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);

    length = atoi(argv[1]);

    /* Dynamically allocating memory for array */
    int *array = new(nothrow) int[length];
    if(!array)
    {
        cout << "Memory allocation for array failed" << endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
        exit(0);
    }

    /* Filling up the array with random data */
    uniform_real_distribution<double> dist(0, 100);
    random_device rd;
    mt19937 mt(rd());
    for (int i = 0; i < length; i++)
        array[i] = dist(mt);

    /* Printing the arrays before exchanging */
    for(int i = 0; i < np; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank)
        {
            if(rank == 0)
                cout << endl << "Before exchanging data: " << endl;

            cout << "Array at process " << rank << ": ";
            for(int j = 0; j < length; j++)
                cout << array[j] << " ";
            cout << endl;
        }
    }

    MPI_Send(array, length, MPI_INT, (rank + 1) % np, 1, MPI_COMM_WORLD);
    MPI_Recv(array, length, MPI_INT, (rank + np - 1) % np, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Printing the arrays after exchanging */
    for(int i = 0; i < np; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank)
        {
            if(rank == 0)
                cout << endl << "After exchanging data: " << endl;
                
            cout << "Array at process " << rank << ": ";
            for(int j = 0; j < length; j++)
                cout << array[j] << " ";
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}