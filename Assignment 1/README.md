# <div align="center">**Parallel Computing - Assignment 1**</div>

# The Algorithm:
* The large matrix is divided into sub-blocks of equal sizes according to the number of processes(Matrix_size / sqrt(num_of_processes)).
* The task of calculating a block is assigned to a block.
* The processes exchange data to calculate the block assigned to it.
* This is repreated and A^N is calculated, where A is matrix and N is the order of the matrix A.
* Finally, each result by the block is written into a file called "product.txt".