#include "matrix_transpo.h"

void matTransposeMpi(const vector<vector<float>> &raw, vector<float> &local_T, int size, int start_row, int num_rows) {
    double start_time = MPI_Wtime();  // Start time for this function
    
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < size; ++j) {
            local_T[i * size + j] = raw[start_row + i][j];
        }
    }
    
    double end_time = MPI_Wtime();  // End time for this function
    double time_taken = end_time - start_time;
    
    if (start_row == 0) {
        FILE* f = fopen("chunkTranspo.csv","a+");
        fprintf(f,"%.f;%lf\n", log2(size), time_taken);
        fclose(f);
    }
}


bool checkSymMpi(const vector<vector<float>> &matrix, int size, int start_row, int num_rows) {
    double start_time = MPI_Wtime();  // Start time for this function
    bool local_isSymmetric = true;

    // Check symmetry in the assigned rows
    for (int i = start_row; i < start_row + num_rows; ++i) {
        for (int j = 0; j < size; ++j) {
            if (matrix[i][j] != matrix[j][i]) {
                local_isSymmetric = false;
                break;
            }
        }
        if (!local_isSymmetric) break;
    }

    // Reduce the results from all processes to determine overall symmetry
    bool global_isSymmetric;
    MPI_Allreduce(&local_isSymmetric, &global_isSymmetric, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();  // End time for this function
    double time_taken = end_time - start_time;
    if (start_row == 0) {
        FILE* f = fopen("chunkCheck.csv","a+");
        fprintf(f,"%.f;%lf\n", log2(size), time_taken);
        fclose(f);
    }
    
    

    return global_isSymmetric;
}