#include "matrix_transpo.h"
#include <omp.h>

void matTransposeOmp(vector<vector<float>> &raw, vector<vector<float>> &transposed,int size)
{
    timespec start, stop;
    clock_gettime( CLOCK_REALTIME, &start);

    #pragma omp parallel for collapse(2)
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            transposed[i][j]=raw[j][i];
        }
    }

    clock_gettime( CLOCK_REALTIME, &stop);

    double timeElapsed=( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) /BILLION;


    char *env_var = getenv("OMP_NUM_THREADS");
    FILE* f = fopen("parallelTranspo.csv","a+");
    fprintf(f,"%.f;%i;%lf\n", log2(size),atoi(env_var), timeElapsed);
    fclose(f);

    //printf("PARALLEL TRANSPOSITION: %lf\n", timeElapsed);
}

bool checkSymOmp(vector<vector<float>> &matrix,int size)
{
    bool isSymmetric = true;
    timespec start, stop;
    clock_gettime( CLOCK_REALTIME, &start);
#pragma omp parallel for collapse(2) reduction(&:isSymmetric)
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            if (matrix[i][j] != matrix[j][i])
                isSymmetric = false;
        }
    }

    clock_gettime( CLOCK_REALTIME, &stop);

    double timeElapsed=( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) /BILLION;
    //printf("PARALLEL CHECK: %lf\n", timeElapsed);

    char *env_var = getenv("OMP_NUM_THREADS");
    FILE* f = fopen("parallelCheck.csv","a+");
    fprintf(f,"%.f;%i;%lf\n", log2(size),atoi(env_var), timeElapsed);
    fclose(f);

    return isSymmetric;
}