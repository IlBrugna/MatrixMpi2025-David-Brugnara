#include "matrix_transpo.h"



#define REPS 8
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int size = pow(2, atoi(argv[1]));
    vector<vector<float>> M(size, vector<float>(size));
    vector<vector<float>> T(size, vector<float>(size));
    
    if (rank == 0) {
        // Initialize matrix M on process 0
        ini_matrix(M, size);
    }

    // Calculate rows per process and handle remainder
    int rows_per_proc = size / num_procs;
    int remainder = size % num_procs;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    vector<float> local_T(num_rows * size);

    for (int i = 0; i < REPS; i++) {
        if (rank==0)
       {
           matTranspose(M,T,size);
           matTransposeOmp(M,T,size);

           checkSym(M,size);
           checkSymOmp(M,size);
       }
    
        // Timer for matTransposeMpi function including communication
        MPI_Barrier(MPI_COMM_WORLD);
        double transpose_start_time = MPI_Wtime();

        // Broadcast the matrix M for transposition
        for (int j = 0; j < size; ++j) {
            MPI_Bcast(M[j].data(), size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Transpose part of the matrix
        matTransposeMpi(M, local_T, size, start_row, num_rows);

        // Gather the transposed parts
        vector<float> recv_buffer;
        if (rank == 0) {
            recv_buffer.resize(size * size);
        }
        
        // Displacements and counts for uneven gather
        vector<int> recv_counts(num_procs);
        vector<int> displacements(num_procs);
        for (int j = 0; j < num_procs; ++j) {
            recv_counts[j] = (rows_per_proc + (j < remainder ? 1 : 0)) * size;
            displacements[j] = (j * rows_per_proc + min(j, remainder)) * size;
        }

        MPI_Gatherv(local_T.data(), num_rows * size, MPI_FLOAT,
                    rank == 0 ? recv_buffer.data() : nullptr, recv_counts.data(), displacements.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        // Reconstruct the transposed matrix T from recv_buffer
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    T[j][i] = recv_buffer[i * size + j];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double transpose_end_time = MPI_Wtime();

        // Timer to have checkSymMpi function include communication
        MPI_Barrier(MPI_COMM_WORLD);
        double sym_check_start_time = MPI_Wtime();

        // Broadcast the matrix M for symmetry check
        for (int j = 0; j < size; ++j) {
            MPI_Bcast(M[j].data(), size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        bool isSymmetric = checkSymMpi(M, size, start_row, num_rows);

        MPI_Barrier(MPI_COMM_WORLD);
        double sym_check_end_time = MPI_Wtime();

        // Output times for each operation
        if (rank == 0) {
            double transpose_total_time = transpose_end_time - transpose_start_time;
            double sym_check_total_time = sym_check_end_time - sym_check_start_time;
            
            FILE* f = fopen("mpiTranspo.csv","a+");
            fprintf(f,"%.f;%i;%lf\n", log2(size),num_procs, transpose_total_time);
            fclose(f);
            
            f = fopen("mpiCheck.csv","a+");
            fprintf(f,"%.f;%i;%lf\n", log2(size),num_procs, sym_check_total_time);
            fclose(f);
        }
    }

    MPI_Finalize();
    return 0;
}



void ini_matrix(vector<vector<float>> &matrix, int size){
    std::random_device rd;
    std::uniform_real_distribution<float> dist(1.f,100.f);
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
           matrix[i][j]=dist(rd); 
        }
    }
}



void print_matrix(vector<vector<float>> &matrix,int size)
{
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            //printf("%i %i value is: %.2f\n",i,j,matrix[i][j]);
            printf("| %.2f |",matrix[i][j]);
        }printf("\n");
    }
}

// Function to verify MPI transposition against sequential transposition
/*
bool verifyTranspose( vector<vector<float>> &originalMatrix, vector<vector<float>> &mpiTransposedMatrix, int size) {
    // Transpose the matrix sequentially
    vector<vector<float>> sequentialTransposed(size, vector<float>(size));
    matTranspose(originalMatrix, sequentialTransposed, size); 

    // Compare the MPI-transposed matrix with the sequentially transposed matrix
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (mpiTransposedMatrix[i][j] != sequentialTransposed[i][j]) {
                return false;
            }
        }
    }
    return true;
}
*/

