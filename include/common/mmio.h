#ifndef __DMMIO_H__
#define __DMMIO_H__

#include <stdint.h>
#include <stdio.h>
#include <string>

#include "macros.h"

/********************* Enums and Data Structures ***************************/

enum MM_VALUE_TYPE {
  MM_VALUE_TYPE_REAL,
  MM_VALUE_TYPE_INTEGER,
  MM_VALUE_TYPE_PATTERN
};

struct Matrix_Metadata {
  // Data type of matrix values
  MM_VALUE_TYPE value_type;
  // Size in bytes of each matrix value
  uint8_t value_bytes;
  bool is_symmetric;
  // First line of the header
  std::string mm_header;
  // Following header lines
  std::string mm_header_body;
};

// This is used for parsing the MTX files
template<typename IT, typename VT>
struct Entry {
  IT row;
  IT col;
  VT val;
};

template<typename IT, typename VT>
struct COO {
  IT nrows;
  IT ncols;
  IT nnz;
  IT *row;
  IT *col;
  VT *val;
};

template<typename IT, typename VT>
struct CSR {
  IT 	nrows;
  IT 	ncols;
  IT 	nnz;
  IT 	*row_ptr;
  IT 	*col_idx;
  VT 	*val;
};

template<typename IT, typename VT>
struct CSC {
  IT 	nrows;
  IT 	ncols;
  IT 	nnz;
  IT 	*col_ptr;
  IT 	*row_idx;
  VT 	*val;
};

/********************* Data Structures Functions ***************************/

/** COO **/
template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

template<typename IT, typename VT>
void DMMIO_COO_destroy(COO<IT, VT>** csr);

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
int DMMIO_COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta);

template<typename IT, typename VT>
int DMMIO_COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta);

// TODO fix
template<typename IT, typename VT>
Entry<IT, VT>* Distr_MMIO_COO_local_read_mpi(FILE *f, bool is_bmtx, int comm_size, int myrank, uint32_t *nentries, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);


/** CSR **/
template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

template<typename IT, typename VT>
void DMMIO_CSR_destroy(CSR<IT, VT>** csr);

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

/** CSC **/
// TODO


/********************* Support Functions ***************************/

char *mm_typecode_to_str(MM_typecode matcode);
int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx);
int mm_read_mtx_crd_size(FILE *f, uint64_t *M, uint64_t *N, uint64_t *nz);
int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[], double val[], MM_typecode matcode);
template<typename IT, typename VT>
int mm_read_mtx_crd_data(FILE *f, int nz, Entry<IT, VT> entries[], MM_typecode matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t value_bytes);

int required_bytes_index(uint64_t maxval);
bool is_file_extension_bmtx(std::string filename);
FILE *open_file_r(const char *filename);

/** BMTX **/
template<typename IT, typename VT>
int write_binary_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta);

#endif // __DMMIO_H__
