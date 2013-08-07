#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include "timer.h"
#include "common.h"


#include "myKernels.h"


// -------------------------------------------------------------------
// Uncomment this to write matrices and vectors to disk in MMX format.
// -------------------------------------------------------------------
#define WRITE_TO_DISK


// -------------------------------------------------------------------
// Utility macros.
// -------------------------------------------------------------------
#define RAND(L,H)  ((L) + ((H)-(L)) * (REAL)rand()/(REAL)RAND_MAX)
#define MAX(A,B)   (((A) > (B)) ? (A) : (B))
#define MIN(A,B)   (((A) < (B)) ? (A) : (B))



// -------------------------------------------------------------------
// GenerateMatrix()
//
// This function generates a banded matrix of size NxN and with
// half-bandwidth K, with non-zero elements between -10 and 10, stored
// on the host in COO format.
// -------------------------------------------------------------------
void
GenerateMatrix(int n, int k, MatCOO_h& Ah)
{
	// Resize the matrix to the appropriate size.
	Ah.resize(n, n, (2 * k + 1) * n - k * (k + 1));

	// Fill in the non-zero elements.
	int iiz = 0;
	for (int ir = 0; ir < n; ir++) {
		int left = MAX(0, ir - k);
		int right = MIN(n - 1, ir + k);

		for (int ic = left; ic <= right; ic++, iiz++) {
			REAL val = RAND(-1.0, 1.0);

			Ah.row_indices[iiz] = ir;
			Ah.column_indices[iiz] = ic;
			Ah.values[iiz] = val;
		}
	}
}


// -------------------------------------------------------------------
// ConvertToBanded()
//
// This function converts the specified host COO matrix to a format
// appropriate for subsequent GPU calculations.
// -------------------------------------------------------------------
void
ConvertToBanded(int n, int k, const MatCOO_h& Ah, Vec_h& Bh)
{
	// Resize the array to the proper length
	Bh.resize((2 * k + 1) * n);

	// Fill in the non-zero entries.
	for (size_t iiz = 0; iiz < Ah.num_entries; iiz++) {
		int ir = Ah.row_indices[iiz];
		int ic = Ah.column_indices[iiz];
		int i = ic * (2 * k + 1) + k + ir - ic;

		Bh[i] = Ah.values[iiz];
	}
}


// -------------------------------------------------------------------
// GenerateRHS()
//
// This function generates the multiple RHS vectors, all stored in a
// contiguous host 1-D array. 
// -------------------------------------------------------------------
void
GenerateRHS(int n, int r, Vec_h& RHSh)
{
	// Resize the array to the proper length.
	RHSh.resize(n * r);
	
	// Fill in the array.
	for (int i = 0; i < n * r; i++)
		RHSh[i] = RAND(-1.0, 1.0);
}


// -------------------------------------------------------------------
// -------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Get matrix size, half-bandwidth, and number of RHS from args.
	if (argc != 4) {
		std::cout << "Usage: multiRHS N K R" << std::endl;
		return 1;
	}

	int n = atoi(argv[1]);
	int k = atoi(argv[2]);
	int r = atoi(argv[3]);


	// Generate the banded matrix on the host in COO format.
	MatCOO_h  Ah;

	GenerateMatrix(n, k, Ah);
#ifdef WRITE_TO_DISK
	cusp::io::write_matrix_market_file(Ah, "A.mtx");
#endif


	// Convert the banded matrix to some appropriate format, stored on the host.
	Vec_h  Bh;

	ConvertToBanded(n, k, Ah, Bh);
#ifdef WRITE_TO_DISK
	cusp::io::write_matrix_market_file(Bh, "B.mtx");
#endif


	// Generate the RHS vectors, all stored in a contiguous
	// 1-D array on the host.
	Vec_h  RHSh;

	GenerateRHS(n, r, RHSh);
#ifdef WRITE_TO_DISK
	cusp::io::write_matrix_market_file(RHSh, "RHS.mtx");
#endif


	// Copy B and RHS to the device.
	Vec  B = Bh;
	Vec  RHS = RHSh;


	// Invoke the function for solving the multiple RHS problem.
	GPUTimer  timer;

	timer.Start();
	SolveL(n, k, r, B, RHS);
	timer.Stop();
	std::cout << "\nForward sweep time: " << timer.getElapsed() << " ms" << std::endl;
#ifdef WRITE_TO_DISK
	cusp::copy(RHS, RHSh);
	cusp::io::write_matrix_market_file(RHSh, "Y.mtx");
#endif

	timer.Start();
	SolveU(n, k, r, B, RHS);
	timer.Stop();
	std::cout << "Backward sweep time: " << timer.getElapsed() << " ms" << std::endl;
#ifdef WRITE_TO_DISK
	cusp::copy(RHS, RHSh);
	cusp::io::write_matrix_market_file(RHSh, "X.mtx");
#endif


	return 0;
}
