/*
 * Author: S. Karthik Chandra 
 * Roll  : CS17B026
 */

#include <mpi.h>
#include <random>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        cerr << "Length of the vector, Number of processes are expected as arguments!\nFormat: mpiexec -np <nproc> ./a.out <length>" << endl;
        exit(0);
    }

    int i, np, rank, length, chunksize;
    double localSum, totalSum = 0;
    double *a;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    length = atoi(argv[1]);
    chunksize = length / np;

    /* MASTER PROCESS */
    if(rank == 0)
    {
        /* Dynamically allocating memory for vectors */
        a = new(nothrow) double[length];
        if(!a)
        {
            cout << "Memory allocation for vectors failed" << endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
            exit(0);
        }

        /* Filling up the vectors with random data */
        uniform_real_distribution<double> dist(0, 100);
        default_random_engine re;
        for (int i = 0; i < length; i++)
            a[i] = dist(re);

        /* Sending "chunksize" number of elements to slave processes */
        int leftover = length % np;
        int offset = chunksize + leftover;
        for(int i = 1; i < np; i++)
        {
            MPI_Send(a + offset, chunksize, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            offset += chunksize;
        }

        /* Master process computes the dot product of "chunksize + leftover" number of elements */
        for(int i = 0; i < chunksize + leftover; i++)
            localSum += (a[i] * a[i]);

        /* Collect local sums from all processes */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        delete[] a;

        cout << "Dot product: " << totalSum << endl;
    }
    /* SLAVE PROCESSES */
    else
    {
        /* Dynamically allocating memory of size "chunksize" */
        a = new double[chunksize];
        MPI_Recv(a, chunksize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Computes local sum */
        for(int i = 0; i < chunksize; i++)
            localSum += (a[i] * a[i]);

        /* Send local sums into master process */
        MPI_Reduce(&localSum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        delete[] a;
    }

    MPI_Finalize();
    return 0;
}