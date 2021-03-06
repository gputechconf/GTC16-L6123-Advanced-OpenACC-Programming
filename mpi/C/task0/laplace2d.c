/*
 *  Copyright 2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENACC
#include <openacc.h>
#endif /*_OPENACC*/
#include "common.h"

//TODO: Include mpi.h
#define N 4096
#define M 4096

float A[N][M];
float Aref[N][M];
float Anew[N][M];

int main(int argc, char** argv)
{
    int iter_max = 1000;
    
    const float pi  = 2.0 * asinf(1.0f);
    const float tol = 1.0e-5f;

    int rank = 0;
    int size = 1;

    memset(A, 0, N * M * sizeof(float));
    memset(Aref, 0, N * M * sizeof(float));
    
    // set boundary conditions
    for (int j = 0; j < N; j++)
    {
        float y0     = sinf( 2.0 * pi * j / (N-1));
        A[j][0]      = y0;
        A[j][M-1]    = y0;
        Aref[j][0]   = y0;
        Aref[j][M-1] = y0;
    }
    
    int jstart = 1;
    int jend   = N-1;

    if ( rank == 0) printf("Jacobi relaxation Calculation: %d x %d mesh\n", N, M);

    if ( rank == 0) printf("Calculate reference solution and time serial execution.\n");
    StartTimer();
    laplace2d_serial( rank, iter_max, tol );
    double runtime_serial = GetTimer();

    if ( rank == 0) printf("Parallel execution.\n");
    StartTimer();
    int iter  = 0;
    float error = 1.0f;
    
    //TODO: Insert OpenACC `data` region
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

        //TODO: Accelerate the next 3 loop nest with OpenACC `kernels` or `parallel`
        for (int j = jstart; j < jend; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
        
        for (int j = jstart; j < jend; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                A[j][i] = Anew[j][i];
            }
        }

        //Periodic boundary conditions
        for( int i = 1; i < M-1; i++ )
        {
                A[0][i]     = A[(N-2)][i];
                A[(N-1)][i] = A[1][i];
        }
        
        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
    double runtime = GetTimer();

    if (check_results( rank, jstart, jend, tol ) && rank == 0)
    {
        printf( "Num GPUs: %d\n", size );
        printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f%\n", N,M, runtime_serial/ 1000.f, size, runtime/ 1000.f, runtime_serial/runtime, runtime_serial/(size*runtime)*100 );
    }
    return 0;
}

#include "laplace2d_serial.h"
