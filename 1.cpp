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

    int np, rank, length, chunksize, localSum = 0, totalSum = 0;
    int *array;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD , &np);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);

    length = atoi(argv[1]);
    chunksize = length / np;
    
    /* MASTER PROCESS */
    if(rank == 0)
    {
        /* Dynamically allocating memory for array */
        array = new(nothrow) int[length];
        if(!array)
        {
            cout << "Memory allocation for array failed" << endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
            exit(0);
        }


        /* Filling up the array with random data */
        uniform_real_distribution<double> dist(0, 100);
        default_random_engine re;
        random_device rd;
		mt19937 mt(rd());
        for (int i = 0; i < length; i++)
            // array[i] = dist(mt);        /* This line is used for random numbers */
            array[i] = dist(re);        /* This line is used for pseudo-random numbers */
        

        /* Sending "chunksize" number of elements to slave processes */
        int leftover = length % np;
        int offset = chunksize + leftover;
        for(int i = 1; i < np; i++)
        {
            MPI_Send(array + offset, chunksize, MPI_INT, i, 1, MPI_COMM_WORLD);
            offset += chunksize;
        }

        /* Master process computes the sum of first "chunksize + leftover" number of elements */
        for(int i = 0; i < chunksize + leftover; i++)
            localSum += array[i];

        /* Collect local sums from all processes */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        delete[] array;

        cout << "Total Sum: " << totalSum << endl;
    }
    /* SLAVE PROCESSES */
    else
    {
        /* Dynamically allocating memory of size "chunksize" */
        array = new int[chunksize];
        MPI_Recv(array, chunksize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Computes local sum */
        for(int i = 0; i < chunksize; i++)
            localSum += array[i];

        /* Send local sums into master process */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        delete[] array;
    }
    
    MPI_Finalize();
    return 0;
}