# README

## Overview

This project runs simulations using `.pbs` scripts for job submission on a cluster. The `.pbs` file used for testing is provided.
## How to Run the Simulation

1. **Copy Files**: Copy all relevant files to your cluster directory.
2. **Specify File Locations**: Update the file paths inside the chosen `.pbs` script to match your cluster directory. 
3. **Adjust Configuration**: If needed modify the enviroment variables in the `.pbs` such as desired queue. 
4. **Submit the Job**:
   - Use the `qsub` command to submit the job.
   - Example: To run the general test, use the command:
     qsub matrix.pbs
# MatrixMpi2025-David-Brugnara
