#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "../../include/common/mmio.h"

// TODO deleteme
#include <ccutils/macros.h>

#define DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template COO<IT, VT>* DMMIO_COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template CSR<IT, VT>* DMMIO_CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template void DMMIO_COO_destroy(COO<IT, VT> **coo); \
  template void DMMIO_CSR_destroy(CSR<IT, VT> **csr); \
  template void DMMIO_Entries_to_COO(Entry<IT, VT> *entries, COO<IT, VT> *coo); \
  template void DMMIO_Entries_to_CSR(Entry<IT, VT> *entries, CSR<IT, VT> *csr); \
  template COO<IT, VT>* DMMIO_COO_read(const char *filename, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta); \
  template CSR<IT, VT>* DMMIO_CSR_read(const char *filename, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta); \
  template COO<IT, VT>* DMMIO_COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta); \
  template CSR<IT, VT>* DMMIO_CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta); \
  template int DMMIO_COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, DMMIO_Matrix_Metadata* meta); \
  template int DMMIO_COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, DMMIO_Matrix_Metadata* meta);


/********************* Utils ***************************/

FILE *open_file_r(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Could not open file [%s] (read).\n", filename);
    return NULL;
  }
  return f;
}

FILE *open_file_w(const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "Could not open file [%s] (write).\n", filename);
    return NULL;
  }
  return f;
}

template<typename IT, typename VT>
int compare_entries(const void *a, const void *b) {
  Entry<IT, VT> *ea = (Entry<IT, VT> *)a;
  Entry<IT, VT> *eb = (Entry<IT, VT> *)b;
  if (ea->row != eb->row)
    return ea->row - eb->row;
  return ea->col - eb->col;
}

/********************* COO ***************************/
template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
  COO<IT, VT> *coo = (COO<IT, VT> *)malloc(sizeof(COO<IT, VT>));
  coo->nrows = nrows;
  coo->ncols = ncols;
  coo->nnz = nnz;
  coo->row = (IT *)malloc(nnz * sizeof(IT));
  coo->col = (IT *)malloc(nnz * sizeof(IT));
  coo->val = NULL;
  if (alloc_val) {
    coo->val = (VT *)malloc(nnz * sizeof(VT));
  }
  return coo;
}

template<typename IT, typename VT>
void DMMIO_COO_destroy(COO<IT, VT> **coo) {
  if (*coo != NULL) {
    if ((*coo)->row != NULL) {
      free((*coo)->row);
      (*coo)->row = NULL;
    }
    if ((*coo)->col != NULL) {
      free((*coo)->col);
      (*coo)->col = NULL;
    }
    if ((*coo)->val != NULL) {
      free((*coo)->val);
      (*coo)->val = NULL;
    }
    free(*coo);
    *coo = NULL;
  }
}

template<typename IT, typename VT>
void DMMIO_Entries_to_COO(Entry<IT, VT> *entries, COO<IT, VT> *coo) {
  for (IT i = 0; i < coo->nnz; ++i) {
    coo->row[i] = entries[i].row;
    coo->col[i] = entries[i].col;
    if (coo->val != NULL) coo->val[i] = entries[i].val;
  }
}

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read(const char *filename, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta) {
  return DMMIO_COO_read_f<IT, VT>(open_file_r(filename), is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
}

template<typename IT, typename VT>
COO<IT, VT>* DMMIO_COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta) {
  IT nrows, ncols, nnz;
  MM_typecode matcode;
  Entry<IT, VT> *entries = mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
  if (entries == NULL) return NULL;
  
  COO<IT, VT> *coo = DMMIO_COO_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
  DMMIO_Entries_to_COO<IT, VT>(entries, coo);
  
  free(entries);

  return coo;
}

template<typename IT, typename VT>
int DMMIO_COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, DMMIO_Matrix_Metadata* meta) {
  return DMMIO_COO_write_f(coo, open_file_w(filename), write_as_binary, meta);
}

template<typename IT, typename VT>
int DMMIO_COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, DMMIO_Matrix_Metadata* meta) {
  if (meta->mm_header.empty()) {
    meta->mm_header = "%%MatrixMarket matrix coordinate";

    switch (meta->value_type) {
      case MM_VALUE_TYPE_REAL:    { meta->mm_header += std::string(MM_REAL_STR);    break; }
      case MM_VALUE_TYPE_INTEGER: { meta->mm_header += std::string(MM_INT_STR);     break; }
      case MM_VALUE_TYPE_PATTERN: { meta->mm_header += std::string(MM_PATTERN_STR); break; }
      default:                  { fprintf(stderr, "BUG: MM_VALUE_TYPE not recognized\n"); return 100; }
    }

    meta->mm_header += meta->is_symmetric ? "symmetric" : "general";
  }
  
  return write_as_binary ? write_binary_matrix_market(f, coo, meta) : write_matrix_market(f, coo, meta);
}


/********************* CSR ***************************/
template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
  CSR<IT, VT> *csr = (CSR<IT, VT> *)malloc(sizeof(CSR<IT, VT>));
  csr->nrows = nrows;
  csr->ncols = ncols;
  csr->nnz = nnz;
  csr->row_ptr = (IT *)malloc((nrows + 1) * sizeof(IT));
  csr->col_idx = (IT *)malloc(nnz * sizeof(IT));
  csr->val = NULL;
  if (alloc_val) {
    csr->val = (VT *)malloc(nnz * sizeof(VT));
  }
  return csr;
}

template<typename IT, typename VT>
void DMMIO_CSR_destroy(CSR<IT, VT> **csr) {
  if (*csr != NULL) {
    if ((*csr)->row_ptr != NULL) {
      free((*csr)->row_ptr);
      (*csr)->row_ptr = NULL;
    }
    if ((*csr)->col_idx != NULL) {
      free((*csr)->col_idx);
      (*csr)->col_idx = NULL;
    }
    if ((*csr)->val != NULL) {
      free((*csr)->val);
      (*csr)->val = NULL;
    }
    free(*csr);
    *csr = NULL;
  }
}

template<typename IT, typename VT>
void DMMIO_Entries_to_CSR(Entry<IT, VT> *entries, CSR<IT, VT> *csr) {
  qsort(entries, csr->nnz, sizeof(Entry<IT, VT>), compare_entries<IT, VT>);

  for (IT v = 0, i = 0; v < csr->nrows; v++) {
    csr->row_ptr[v] = i;
    while (i < csr->nnz && entries[i].row == v) {
      csr->col_idx[i] = entries[i].col;
      if (csr->val != NULL) {
        csr->val[i] = entries[i].val;
      }
      ++i;
    }
  }
  csr->row_ptr[csr->nrows] = csr->nnz;
}

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read(const char *filename, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta) {
  return DMMIO_CSR_read_f<IT, VT>(open_file_r(filename), is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
}

template<typename IT, typename VT>
CSR<IT, VT>* DMMIO_CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, DMMIO_Matrix_Metadata* meta) {
  IT nrows, ncols, nnz;
  MM_typecode matcode;
  Entry<IT, VT> *entries = mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
  if (entries == NULL) return NULL;

  CSR<IT, VT> *csr = DMMIO_CSR_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
  DMMIO_Entries_to_CSR<IT, VT>(entries, csr);

  free(entries);

  return csr;
}


DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_STRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)