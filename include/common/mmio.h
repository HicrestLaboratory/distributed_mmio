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

/// @brief IMPORTANT: initialize with "new"
struct DMMIO_Matrix_Metadata {
  // Data type of matrix values
  MM_VALUE_TYPE value_type;
  // Size in bytes of each matrix value
  uint8_t value_bytes;
  // Size in bytes of each matrix index
  uint8_t index_bytes;
  // First line of the header
  std::string mm_header;
  // Following header lines
  std::string mm_header_body;
  bool is_symmetric;
  bool is_pattern;
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
void DMMIO_COO_destroy(COO<IT, VT>** coo);

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read(const char *filename, bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
int DMMIO_COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, DMMIO_Matrix_Metadata* meta);

template<typename IT, typename VT>
int DMMIO_COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, DMMIO_Matrix_Metadata* meta);

/** CSR **/
template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

template<typename IT, typename VT>
void DMMIO_CSR_destroy(CSR<IT, VT>** csr);

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read(const char *filename, bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL);

/** CSC **/
// TODO


/********************* Support Functions ***************************/

/// @brief Serializes MM_typecode
/// @param matcode 
/// @return Serialized `matcode`
char *mm_typecode_to_str(MM_typecode matcode);

/// @brief Read the first line of the MTX file and populates `matcode`
/// @param f The file to read from 
/// @param matcode 
/// @param is_bmtx If `true`, the function expects to find sizes (in bytes) for indices and values
/// @param meta
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta);

/// @brief Reads and parses the MTX line containing matrix `nrows ncols nnz`
/// @param f The file to read from
/// @param M Number of rows will be stored here
/// @param N Number of columns will be stored here
/// @param nz Number of non-zeros will be stored here
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
int mm_read_mtx_crd_size(FILE *f, uint64_t *M, uint64_t *N, uint64_t *nz);

/// @brief Reads and parses entries (i.e. MTX couples or triples)
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param f The file to read from
/// @param nentries Number of entries in the file (not necessarily the number of non-zeros)
/// @param entries Entries will be written here
/// @param matcode 
/// @param is_bmtx 
/// @param idx_bytes 
/// @param value_bytes 
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
template<typename IT, typename VT>
int mm_read_mtx_crd_data(FILE *f, int nentries, Entry<IT, VT> entries[], MM_typecode *matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t value_bytes);

/// @brief Reads and parses the header including matrix shape (nrows, ncols, nnz)
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param f The file to read from
/// @param nrows 
/// @param ncols 
/// @param nentries number of entries in the MTX file 
/// @param nnz_upperbound This is not equal in the case of symmetric matrices that have entries on the diagonal
/// @param matcode 
/// @param is_bmtx 
/// @param meta 
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
template<typename IT, typename VT>
int mm_parse_header(FILE *f, IT &nrows, IT &ncols, IT &nentries, IT &nnz_upperbound, MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta);

/// @brief Complete "non-distributed" MTX parsing. Reads and parses the header and (raw) entries 
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param f The file to read from
/// @param nrows 
/// @param ncols 
/// @param nnz 
/// @param matcode 
/// @param is_bmtx 
/// @param meta 
/// @return An "heap-allocated" array of entries. Not sorted (retain MTX file entries ordering). In case of symmetric matrices, entries will be explicitly duplicated.
template<typename IT, typename VT>
Entry<IT, VT>* mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta);

/// @brief If the matrix is symmetric (from `meta`) duplicate entries. This wont duplicate entries on the diagonal.
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param entries 
/// @param nentries 
/// @param meta 
/// @return The actual number of non-zeros
template<typename IT, typename VT>
IT mm_duplicate_entries_for_symmetric_matrices(Entry<IT, VT>* entries, IT nentries, DMMIO_Matrix_Metadata* meta);

/// @brief Sets `meta` values form `matcode`
/// @param meta 
/// @param matcode 
void mm_set_metadata(DMMIO_Matrix_Metadata* meta, MM_typecode *matcode);

/// @brief Writes COO to MTX file
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param f The file to write to
/// @param coo 
/// @param meta Information to construct the header
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
template<typename IT, typename VT>
int write_matrix_market(FILE *f, COO<IT, VT> *coo, DMMIO_Matrix_Metadata *meta);

/// @brief Computes the minimum number of bytes required to store indices that range from `0` to `maxval` (using `unsigned integers`)  
/// @param maxval 
/// @return The number of bytes
int required_bytes(uint64_t maxval);

/// @brief Opens a file in read-only mode
/// @param filename 
/// @return 
FILE *open_file_r(const char *filename);

/** BMTX **/

/// @brief Detect if file is `bmtx` from file extension
/// @param filename 
/// @return `true` if `filename` ends with `.bmtx`, otherwise `false`
bool is_file_extension_bmtx(std::string filename);

/// @brief Writes COO to a `bmtx` file (in binary format)
/// @tparam IT Index Data Type
/// @tparam VT Value Data Type
/// @param f File to write to
/// @param coo 
/// @param meta Information to construct the header
/// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
template<typename IT, typename VT>
int write_binary_matrix_market(FILE *f, COO<IT, VT> *coo, DMMIO_Matrix_Metadata *meta);

#endif // __DMMIO_H__
