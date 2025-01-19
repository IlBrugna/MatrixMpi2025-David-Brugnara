#include "matrix_transpo.h"

void matTranspose(vector<vector<float>> &raw, vector<vector<float>> &transposed,int size)
{
    timespec start, stop;
    clock_gettime( CLOCK_REALTIME, &start);

    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            transposed[i][j]=raw[j][i];
        }
    }
    clock_gettime( CLOCK_REALTIME, &stop);

    double timeElapsed=( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) /BILLION;

    FILE* f = fopen("sequentialTranspo.csv","a+");
    fprintf(f,"%.f;%lf\n", log2(size), timeElapsed);
    fclose(f);

    //printf("SEQUENTIAL TRANSPOSITION: %lf\n", timeElapsed);
}

bool checkSym(vector<vector<float>> &matrix,int size)
{
    bool isSymmetric = true;
    timespec start, stop;
    clock_gettime( CLOCK_REALTIME, &start);

    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            if (matrix[i][j] != matrix[j][i]) isSymmetric = false;
        }
    }

    clock_gettime( CLOCK_REALTIME, &stop);

    double timeElapsed=( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) /BILLION;
    //printf("SEQUENTIAL CHECK: %lf\n", timeElapsed);

    FILE* f = fopen("sequentialCheck.csv","a+");
    fprintf(f,"%.f;%lf\n", log2(size), timeElapsed);
    fclose(f);

    return isSymmetric;
}