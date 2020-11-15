# <div align="center">**Parallel Computing - Assignment 3**</div>
## Instructions for running:
1. Question 1:

        mpic++ 1.cpp
        mpiexec -np <proc> ./a.out <size>

2. Question 2:

        mpic++ 2.cpp
        mpiexec -np <proc> ./a.out <size>

3. Question 3:

        mpic++ 3.cpp
        mpiexec -np <proc> ./a.out <size>

4. Question 4:
            
            -

5. Question 5:

        g++ 5.cpp
        ./a.out <nthreads> <size>

6. Question 6:

        g++ -fopenmp 6.cpp
        ./a.out <size> <nthreads>

7. Question 7:

        mpic++ -fopenmp 7.cpp
        mpiexec -np <nproc> ./a.out <size> <nthreads>

8. Question 8:

        mpic++ 8.cpp
        mpiexec -np <nproc> ./a.out <length>

9. Question 9:

        g++ -fopenmp 9.cpp
        ./a.out <m> <n> <p> <nthreads>

10. Question 10:

        mpic++ 10.cpp
        mpiexec -np <nproc> ./a.out <size> <iterations>

11. Question 11:

        mpic++ 11.cpp -lpthread
        mpiexec -np <nproc> ./a.out <nthreads>