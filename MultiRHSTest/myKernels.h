// -------------------------------------------------------------------
// Sample header file implementing the function SolveL() and SolveU()
// expected by the driver program.
// -------------------------------------------------------------------

#ifndef MULTI_RHS_KERNELS_H
#define MULTI_RHS_KERNELS_H


// Header file with problem dimensions and typedefs.
#include "common.h"


// -------------------------------------------------------------------
// Forward declarations for CUDA kernels.
// -------------------------------------------------------------------
__global__ void  forward_kernel(int n, int k, int r, REAL* B, REAL* RHS);
__global__ void  backward_kernel(int n, int k, int r, REAL* B, REAL* RHS);


// -------------------------------------------------------------------
// SolveL()
// SolveU()
//
// Sample implementations of the two solve functions invoked by the
// driver program for the solution of multiple RHS problem.
//
// The device vector B is assumed to encode the L and U factors as
// they would be obtained after an LU factorization of an n-by-n
// banded matrix with half-bandwidth k.
//
// The function SolveL() should overwrite RHS with the solution of
// L * X = RHS, for all r vectors encoded in RHS.
//
// The function SolveU() should overwrite RHS with the solution of
// U * X = RHS, for all r vectors encoded in RHS.
// -------------------------------------------------------------------
void
SolveL(int n, int k, int r, Vec& B, Vec& RHS)
{
	// Get access to the underlying pointers in B and RHS.
	REAL* p_B = thrust::raw_pointer_cast(&B[0]);
	REAL* p_RHS = thrust::raw_pointer_cast(&RHS[0]);
    
	// Invoke the CUDA kernel to do the work.
    //	forward_kernel<<<r, n>>>(n, k, r, p_B, p_RHS);
	forward_kernel<<<1, r>>>(n, k, r, p_B, p_RHS);
}

void
SolveU(int n, int k, int r, Vec& B, Vec& RHS)
{
	// Get access to the underlying pointers in B and RHS.
	REAL* p_B = thrust::raw_pointer_cast(&B[0]);
	REAL* p_RHS = thrust::raw_pointer_cast(&RHS[0]);
    
	// Invoke the CUDA kernel to do the work.
    //	backward_kernel<<<r, n>>>(n, k, r, p_B, p_RHS);
	backward_kernel<<<1, r>>>(n, k, r, p_B, p_RHS);
}


// -------------------------------------------------------------------
// forward_kernel()
// backward_kernel()
//
// These are the CUDA kernels for performing the forward and backward
// elimination sweeps for solving the problem (L*U)*X = RHS. It is
// assumed that the eliminations are done "in-place".
// -------------------------------------------------------------------
__global__ void
    
   int bid = blockIdx.x + (gridDim.x * blockIdx.y);
   int tid = threadIdx.x  + (blockDim.x * threadIdx.y);
   int gid = bid * (blockDim.x * blockDim.y) + tid; 

   int l = 0; 

   for (int i = n * gid + 1 ;  i < (n * gid) + n; i ++, l++) {
	//printf("\nRHS[%d] = RHS[%d]",i , i);
	int m = 0;
	for (int j = i-k; j < i; j++, m++) {
		//if (m > 0) {
		//printf("\n(l-(k-2))*(2*k+1)+(m)*(2*k)-1 = %d", (l-(k-2))*(2*k+1)+(m)*(2*k)-1);
		//printf("\n(m)*(2*k)-1 = %d", (m)*(2*k)-1);
		if ((l-(k-2))*(2*k+1)+(m)*(2*k)-1 > k ) {
			//printf("\n\tRHS[%d] = %f", j, RHS[j]);
			//printf("\n\tB[%d] = %f", (l-1)*(2*k+1)+(m)*(2*k)-1, B[(l-1)*(2*k+1)+(m)*(2*k)-1]);
			//printf("\n\tB[%d] = %f", (l-(k-2))*(2*k+1)+(m)*(2*k)-1, B[(l-(k-2))*(2*k+1)+(m)*(2*k)-1]);
			//printf("\n\tB[%d] = %f", (l+1)*(2*k+1)-1, B[(l+1)*(2*k+1)-1]);
			//printf(" - B[%d]*RHS[%d]", (l-1)*(2*k+1)+(m)*(2*k)-1, j);
			//printf("\nB[%d] = %f, l = %d, m = %d", 
			//	(l-1)*(2*k+1)+(m)*(2*k)-1, B[(l-1)*(2*k+1)+(m)*(2*k)-1], l, m);
			//RHS[i] = RHS[i] - B[(l+1)*(2*k+1)-1]*RHS[j];
			//RHS[i] = RHS[i] - B[(l-1)*(2*k+1)+(m)*(2*k)-1]*RHS[j];
			RHS[i] = RHS[i] - B[(l-(k-2))*(2*k+1)+(m)*(2*k)-1]*RHS[j];
		}
	}
	//printf("\n");
   }
}
__global__ void
backward_kernel(int n, int k, int r, REAL* B, REAL* RHS)
{
	
   int bid = blockIdx.x + (gridDim.x * blockIdx.y);
   int tid = threadIdx.x  + (blockDim.x * threadIdx.y);
   int gid = bid * (blockDim.x * blockDim.y) + tid; 
   
   int l = n-1;
   
   // do the first iteration outside the loop for indexing ease 
   RHS[n * gid + n - 1] = RHS[n * gid + n - 1] / B[l*(2*k+1)+1];
   
   l -= 1;
   // printf("\nl = %d", l);
   
   for (int i = n * gid + n -  2 ;  i > n*gid - 1; i--, l--) {
	for (int j = i+1; j < i + k + 1; j++) {       
		//printf("\n\t\tAccessing index %d of RHS", i);
		//printf("\n\t\tAccessing index %d of RHS", j);
		//printf("\n\t\tAccessing index %d of B",(l+1)*(2*k+1));
		RHS[i] = RHS[i] - B[(l+1)*(2*k+1)]*RHS[j]; 
	}
	RHS[i] = RHS[i] / B[l*(2*k+1)+1];
	//printf("\n\tAccessing index %d of B", l*(2*k+1)+1);
	//printf("\n");
    }
}

#endif
