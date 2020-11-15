/*
 *  Author: S. Karthik Chandra
 *  Roll  : CS17B026
 */

#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        cerr << "Size of the array, Number of iterations are expected as arguments!\nFormat: mpiexec -np <nproc> ./a.out <size> <iterations>" << endl;
        exit(0);
    }
    int length = atoi(argv[1]);
    int m = atoi(argv[2]);

    int rank, np;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* Dynamically allocate the arrays */
    int *array = new int[length];
    if(!array)
    {
        cerr << "Memory allocation for array failed" << endl;
        exit(0);
    }

    /* Initializing the arrays and printing them(values before 'm' iterations) */
    for (int j = 0; j < np; j++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == j)
        {
            if(rank == 0)
                cout << endl << "Before exchanging data: " << endl;
            
            cout << "Rank " <<  rank << ": ";
            for (int i = 0; i < length; i++)
            {
                array[i] = rank;
                cout << array[i] << ' ';
            }
            cout << "\n";
        }
    }

    /* The main data exchanging part */
    int X = 0, new_X;
    for (int i = 0; i < m; i++)
    {
        if (X == rank)
            new_X = rand() % np;

        /* Let everyone know the new_X */ 
        MPI_Bcast(&new_X, 1, MPI_INT, X, MPI_COMM_WORLD);

        /* If the new process is same as the old one, continue */
        if (new_X == X)
            continue;

        /* Current X sends the new_X array values */
        if (X == rank)
            MPI_Send(&array[0], length, MPI_DOUBLE, new_X, 1, MPI_COMM_WORLD);

        /* New_x receives the array from current X */
        if (rank == new_X)
            MPI_Recv(&array[0], length, MPI_DOUBLE, X, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        X = new_X;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Printing output after 'm' iterations */
    for (int j = 0; j < np; j++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == j)
        {
            if(rank == 0)
                cout << endl << "After exchanging data: " << endl;
 
            cout << "Rank " <<  rank << ": ";
            for (int i = 0; i < length; i++)
                cout << array[i] << ' ';
            cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}