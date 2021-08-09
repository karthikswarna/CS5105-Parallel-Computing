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

    int np, rank, length, offset, chunksize, localSum = 0, totalSum = 0;
    int *array;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD , &np);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);

    length = atoi(argv[1]);
    chunksize = length / np;
    
    /* MASTER PROCESS */
    if(rank == 0)
    {
        /* Sending offsets to slave processes */
        int leftover = length % np;
        offset = chunksize + leftover;
        for(int i = 1; i < np; i++)
        {
            MPI_Send(&offset, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            offset += chunksize;
        }

        /* Master process computes the sum of first "chunksize + leftover" elements */
        for(int i = 0; i < chunksize + leftover; i++)
            localSum += i;

        /* Collect local sums from all processes */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        cout << "Total Sum: " << totalSum << endl;
    }
    /* SLAVE PROCESSES */
    else
    {
        /* Receives the offset from master process */
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Computes local sum */
        for(int i = offset; i < offset + chunksize; i++)
            localSum += i;

        /* Send local sums into master process */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}