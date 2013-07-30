#ifndef MULTI_RHS_COMMON_H
#define MULTI_RHS_COMMON_H


#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>


// -------------------------------------------------------------------
// Various type definitions for readibility.
// -------------------------------------------------------------------
typedef double REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> MatCSR;
typedef typename cusp::coo_matrix<int, REAL, cusp::device_memory> MatCOO;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vec;
typedef typename cusp::array1d<int, cusp::device_memory>          VecI;

typedef typename cusp::csr_matrix<int, REAL, cusp::host_memory>   MatCSR_h;
typedef typename cusp::coo_matrix<int, REAL, cusp::host_memory>   MatCOO_h;
typedef typename cusp::array1d<REAL, cusp::host_memory>           Vec_h;
typedef typename cusp::array1d<int, cusp::host_memory>            VecI_h;


// -------------------------------------------------------------------
// Declarations of global functions.
// -------------------------------------------------------------------
void SolveL(Vec& B, Vec& RHS);
void SolveU(Vec& B, Vec& RHS);


#endif