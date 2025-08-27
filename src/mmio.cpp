/*
 *   Matrix Market I/O library for ANSI C
 *   See http://math.nist.gov/MatrixMarket for details.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "../include/mmio.h"
#include "../../ccutils/include/ccutils/colors.h"
#include "../../ccutils/include/ccutils/macros.h"
#include <unistd.h>

#define MMIO_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template int mm_read_mtx_crd_data(FILE *f, int nnz, Entry<IT, VT> *entries, MM_typecode matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t val_bytes); \
  template CSR_local<IT, VT>* Distr_MMIO_CSR_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template void Distr_MMIO_CSR_local_destroy(CSR_local<IT, VT> **csr); \
  template COO_local<IT, VT>* Distr_MMIO_COO_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template void Distr_MMIO_COO_local_destroy(COO_local<IT, VT> **coo); \
  template void entries_to_local_csr(Entry<IT, VT> *entries, CSR_local<IT, VT> *csr); \
  template void entries_to_local_coo(Entry<IT, VT> *entries, COO_local<IT, VT> *coo); \
  template int write_binary_matrix_market(FILE *f, COO_local<IT, VT> *coo, Matrix_Metadata *meta); \
  template CSR_local<IT, VT>* Distr_MMIO_CSR_local_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template CSR_local<IT, VT>* Distr_MMIO_CSR_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template COO_local<IT, VT>* Distr_MMIO_COO_local_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template COO_local<IT, VT>* Distr_MMIO_COO_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template Entry<IT, VT>* Distr_MMIO_COO_local_read_mpi(FILE *f, bool is_bmtx, int comm_size, int myrank, uint32_t* nentries, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template int Distr_MMIO_COO_local_write(COO_local<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta); \
  template int Distr_MMIO_COO_local_write_f(COO_local<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta); \
  template Entry<IT, VT>* sortEntriesByOwner(const Entry<IT, VT>* entries, const int* owner, size_t nentries);

  // template Entry<IT, VT>* mm_parse_file(FILE *f);
  // template int compare_entries_csr(const void *a, const void *b);

/**
 * Matrix Market parsing utilities
 */ 

int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta) {
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH];
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  uint8_t idx_bytes, val_bytes;
  char *p;

  mm_clear_typecode(matcode);

  if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
    return MM_PREMATURE_EOF;

  if (meta) {
    meta->mm_header = std::string(line);
    if (!meta->mm_header.empty() && meta->mm_header.back() == '\n') {
      meta->mm_header.pop_back();
    }
  }

  if (is_bmtx) {
    if (sscanf(line, "%s %s %s %s %s %hhu %hhu", banner, mtx, crd, data_type, storage_scheme, &idx_bytes, &val_bytes) != 7)
      return MM_PREMATURE_EOF;
    mm_set_idx_bytes(matcode, idx_bytes);
    mm_set_val_bytes(matcode, val_bytes);    
  } else {
    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
      return MM_PREMATURE_EOF;
  }

  for (p = mtx; *p != '\0'; *p = tolower(*p), p++)
    ; /* convert to lower case */
  for (p = crd; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = data_type; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++)
    ;

  /* check for banner */
  if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    return MM_NO_HEADER;

  /* first field should be "mtx" */
  if (strcmp(mtx, MM_MTX_STR) != 0)
    return MM_UNSUPPORTED_TYPE;
  mm_set_matrix(matcode);

  /* second field describes whether this is a sparse matrix (in coordinate
          storgae) or a dense array */

  if (strcmp(crd, MM_SPARSE_STR) == 0)
    mm_set_sparse(matcode);
  else if (strcmp(crd, MM_DENSE_STR) == 0)
    mm_set_dense(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* third field */

  if (strcmp(data_type, MM_REAL_STR) == 0)
    mm_set_real(matcode);
  else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
    mm_set_complex(matcode);
  else if (strcmp(data_type, MM_PATTERN_STR) == 0)
    mm_set_pattern(matcode);
  else if (strcmp(data_type, MM_INT_STR) == 0)
    mm_set_integer(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* fourth field */

  if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
    mm_set_general(matcode);
  else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
    mm_set_symmetric(matcode);
  else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
    mm_set_hermitian(matcode);
  else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
    mm_set_skew(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  // Read and store all header lines (starting with '%')
  if (meta) {
    long pos = ftell(f);
    while (fgets(line, sizeof(line), f)) {
      if (line[0] != '%') {
        fseek(f, pos, SEEK_SET); // back to start of non-comment line
        break;
      }
      meta->mm_header_body += line;
      pos = ftell(f);
    }
  }

  return 0;
}

int mm_read_mtx_crd_size(FILE *f, uint64_t *nrows, uint64_t *ncols, uint64_t *nnz) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;

  /* set return null parameter values, in case we exit with errors */
  *nrows = *ncols = *nnz = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  /* line[] is either blank or has nrows,ncols, nnz */
  if (sscanf(line, "%lu %lu %lu", nrows, ncols, nnz) == 3)
    return 0;

  else
    do {
      num_items_read = fscanf(f, "%lu %lu %lu", nrows, ncols, nnz);
      if (num_items_read == EOF)
        return MM_PREMATURE_EOF;
    } while (num_items_read != 3);

  return 0;
}

// FIXME this is a draft
// template<typename IT, typename VT>
// int parse_ascii_entries(FILE *f, int nentries, Entry<IT, VT> *entries, MM_typecode matcode) {
//   // Heuristically assume a line is ~50–80 chars
//   const size_t buf_size = nentries * 64;
//   char *buffer = (char *)malloc(buf_size);
//   if (!buffer) {
//     fprintf(stderr, "Failed to allocate %zu bytes for ASCII read buffer.\n", buf_size);
//     return MM_COULD_NOT_READ_FILE;
//   }

//   size_t read = fread(buffer, 1, buf_size, f);
//   if (read == 0) {
//     free(buffer);
//     return MM_PREMATURE_EOF;
//   }

//   char *ptr = buffer;
//   char *end = buffer + read;
//   int count = 0;
//   bool is_pattern = mm_is_pattern(matcode);
//   bool is_real_or_int = mm_is_real(matcode) || mm_is_integer(matcode);

//   while (ptr < end && count < nentries) {
//     // Skip whitespace
//     while (ptr < end && std::isspace(*ptr)) ++ptr;
//     if (ptr >= end) break;

//     // Parse row
//     IT row = static_cast<IT>(strtoull(ptr, &ptr, 10)) - 1;

//     // Parse col
//     while (ptr < end && std::isspace(*ptr)) ++ptr;
//     IT col = static_cast<IT>(strtoull(ptr, &ptr, 10)) - 1;

//     VT val = static_cast<VT>(1.0); // default for pattern

//     if (is_real_or_int) {
//       while (ptr < end && std::isspace(*ptr)) ++ptr;
//       val = static_cast<VT>(strtod(ptr, &ptr));
//     }

//     entries[count].row = row;
//     entries[count].col = col;
//     entries[count].val = val;
//     ++count;
//   }

//   free(buffer);

//   return (count == nentries) ? 0 : MM_PREMATURE_EOF;
// }

template<typename IT, typename VT>
int mm_read_mtx_crd_data(FILE *f, int nentries, Entry<IT, VT> *entries, MM_typecode matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t val_bytes) {
  bool is_pattern = mm_is_pattern(matcode);
  
  size_t entry_size = 2 * idx_bytes + (is_pattern ? 0 : val_bytes);
  size_t total_size = nentries * entry_size;

  if (!is_bmtx) {
    // FIXME uncomment and test return parse_ascii_entries(f, nentries, entries, matcode);

    // Original ASCII Matrix Market parsing
    const char *I_FMT = std::is_same<IT, uint64_t>::value ? "%lu" : "%u";
    const char *V_FMT = std::is_same<VT, double>::value   ? "%lg" : "%g";
    char fmt[32];
    int i;

    if (mm_is_real(matcode) || mm_is_integer(matcode)) {
      snprintf(fmt, 32, "%s %s %s", I_FMT, I_FMT, V_FMT);
      for (i = 0; i < nentries; i++) {
        if (fscanf(f, fmt, &entries[i].row, &entries[i].col, &entries[i].val) != 3)
          return MM_PREMATURE_EOF;
        --entries[i].row;
        --entries[i].col;
      }
    } else if (is_pattern) {
      snprintf(fmt, sizeof(fmt), "%s %s", I_FMT, I_FMT);
      for (i = 0; i < nentries; i++) {
        if (fscanf(f, fmt, &entries[i].row, &entries[i].col) != 2)
          return MM_PREMATURE_EOF;
        --entries[i].row;
        --entries[i].col;
        entries[i].val = static_cast<VT>(1.0);
      }
    } else return MM_UNSUPPORTED_TYPE;

    return 0;
  }

  // Binary BMTX parsing
  
  // Allocate buffer to read the entire data block
  uint8_t *buffer = (uint8_t *)malloc(total_size);
  if (!buffer) {
    fprintf(stderr, "Failed to allocate %zu bytes for input buffer.\n", total_size);
    return MM_COULD_NOT_READ_FILE;
  }

  if (fread(buffer, 1, total_size, f) != total_size) {
    fprintf(stderr, "Failed to read expected %zu bytes from file.\n", total_size);
    free(buffer);
    return MM_PREMATURE_EOF;
  }

  uint8_t *ptr = buffer;

  for (int i = 0; i < nentries; ++i) {
    uint64_t row = 0, col = 0;

    // Read row
    memcpy(&row, ptr, idx_bytes);
    entries[i].row = static_cast<IT>(row);
    ptr += idx_bytes;

    // Read col
    memcpy(&col, ptr, idx_bytes);
    entries[i].col = static_cast<IT>(col);
    ptr += idx_bytes;

    // Read val if present
    if (!is_pattern) {
      if (val_bytes == 4) {
        float val_f;
        memcpy(&val_f, ptr, sizeof(float));
        entries[i].val = static_cast<VT>(val_f);
      } else if (val_bytes == 8) {
        double val_d;
        memcpy(&val_d, ptr, sizeof(double));
        entries[i].val = static_cast<VT>(val_d);
      } else {
        free(buffer);
        return MM_UNSUPPORTED_TYPE;
      }
      ptr += val_bytes;
    } else {
      entries[i].val = static_cast<VT>(1.0);  // Default for pattern
    }
  }

  free(buffer);
  return 0;
}


int required_bytes_index(uint64_t maxval) {
  if (maxval <= UINT8_MAX)  return 1;
  if (maxval <= UINT16_MAX) return 2;
  if (maxval <= UINT32_MAX) return 4;
  return 8;
}

/**
 * Structs constructors and destructors
 */ 

// CSR

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
  CSR_local<IT, VT> *csr = (CSR_local<IT, VT> *)malloc(sizeof(CSR_local<IT, VT>));
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
void Distr_MMIO_CSR_local_destroy(CSR_local<IT, VT> **csr) {
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

// COO

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
  COO_local<IT, VT> *coo = (COO_local<IT, VT> *)malloc(sizeof(COO_local<IT, VT>));
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
void Distr_MMIO_COO_local_destroy(COO_local<IT, VT> **coo) {
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

/**
 * Specific parsing functions
 */ 

// CSR

template<typename IT, typename VT>
int compare_entries_csr(const void *a, const void *b) {
  Entry<IT, VT> *ea = (Entry<IT, VT> *)a;
  Entry<IT, VT> *eb = (Entry<IT, VT> *)b;
  if (ea->row != eb->row)
    return ea->row - eb->row;
  return ea->col - eb->col;
}

template<typename IT, typename VT>
void entries_to_local_csr(Entry<IT, VT> *entries, CSR_local<IT, VT> *csr) {
  qsort(entries, csr->nnz, sizeof(Entry<IT, VT>), compare_entries_csr<IT, VT>);

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

// COO

template<typename IT, typename VT>
void entries_to_local_coo(Entry<IT, VT> *entries, COO_local<IT, VT> *coo) {
  for (IT i = 0; i < coo->nnz; ++i) {
    coo->row[i] = entries[i].row;
    coo->col[i] = entries[i].col;
    if (coo->val != NULL) coo->val[i] = entries[i].val;
  }
}

/**
 * Read functions
 */ 

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

bool is_file_extension_bmtx(std::string filename) {
  return filename.size() >= 5 && filename.compare(filename.size() - 5, 5, ".bmtx") == 0;
}

int write_matrix_market_header(FILE *f, Matrix_Metadata *meta, int index_bytes, uint64_t nrows, uint64_t ncols, uint64_t nentries) {
  if (!f) return MM_COULD_NOT_WRITE_FILE;
  
  std::string header = meta->mm_header;
  // Strip trailing spaces
  while (!header.empty() && header.back() == ' ') header.pop_back();
  int space_count = std::count(header.begin(), header.end(), ' ');
  if (space_count >= 4) {
    // Find position of 5th space (start of 6th token)
    size_t pos = 0;
    int count = 0;
    while (count < 5 && pos != std::string::npos) {
      pos = header.find(' ', pos);
      if (pos != std::string::npos) {
        ++count;
        ++pos;
      }
    }
    if (pos != std::string::npos) {
      header = header.substr(0, pos); // up to and including 5th space
    }
  }
  if (index_bytes > 0) { // This determines if header is for bmtx
    header += " " + std::to_string(index_bytes) + " " + std::to_string(meta->val_bytes > 0 ? meta->val_bytes : 4);
  }
  fprintf(f, "%s\n", header.c_str());
  if (!meta->mm_header_body.empty()) {
    std::string header_body = meta->mm_header_body;
    // Remove all trailing newlines to ensure only one \n after print
    while (!header_body.empty() && header_body.back() == '\n') {
      header_body.pop_back();
    }
    fprintf(f, "%s\n", header_body.c_str());
  }
  // Write size line
  fprintf(f, "%ld %ld %ld\n", nrows, ncols, nentries);
  return 0;
}

template<typename IT, typename VT>
int write_binary_matrix_market(FILE *f, COO_local<IT, VT> *coo, Matrix_Metadata *meta) {
  if (!f) return MM_COULD_NOT_WRITE_FILE;

  int index_bytes = required_bytes_index(std::max(coo->nrows, coo->ncols));

  IT nentries = coo->nnz;
  if (meta->is_symmetric) { // TODO optimize
    nentries = 0;
    for (IT i = 0; i < coo->nnz; ++i) {
      if (coo->row[i] <= coo->col[i]) {
        ++nentries;
      }
    }
  }

  int err = write_matrix_market_header(f, meta, index_bytes, coo->nrows, coo->ncols, nentries);
  if (err != 0) {
    fprintf(stderr, "Something went wrong writing the file header.\n");
    fclose(f);
    return err;
  }

  // Write binary data
  for (IT i = 0; i < coo->nnz; ++i) {
    if (meta->is_symmetric && coo->row[i] > coo->col[i]) continue; // For patter matrices

    // Write row
    switch (index_bytes) {
      case 1: { uint8_t  v = (uint8_t)coo->row[i];  fwrite(&v, sizeof(v), 1, f); break; }
      case 2: { uint16_t v = (uint16_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
      case 4: { uint32_t v = (uint32_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
      case 8: { uint64_t v = (uint64_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
    }

    // Write column
    switch (index_bytes) {
      case 1: { uint8_t  v = (uint8_t)coo->col[i];  fwrite(&v, sizeof(v), 1, f); break; }
      case 2: { uint16_t v = (uint16_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
      case 4: { uint32_t v = (uint32_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
      case 8: { uint64_t v = (uint64_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
    }

    // Write value
    if (meta->val_type != MM_VAL_TYPE_PATTERN && coo->val != NULL) {
      // TODO generalize
      if (meta->val_bytes == 8) { double v = (double)coo->val[i]; fwrite(&v, sizeof(double), 1, f); }
      else                      { float  v =  (float)coo->val[i]; fwrite(&v, sizeof(float),  1, f); }
    }
  }

  fclose(f);
  return 0;
}

template<typename IT, typename VT>
int write_matrix_market(FILE *f, COO_local<IT, VT> *coo, Matrix_Metadata *meta) {
  if (!f) return MM_COULD_NOT_WRITE_FILE;

  IT nentries = coo->nnz;
  if (meta->is_symmetric) { // TODO optimize
    nentries = 0;
    for (IT i = 0; i < coo->nnz; ++i) {
      if (coo->row[i] >= coo->col[i]) {
        ++nentries;
      }
    }
  }

  int err = write_matrix_market_header(f, meta, -1, coo->nrows, coo->ncols, nentries);
  if (err != 0) {
    fprintf(stderr, "Something went wrong writing the file header.\n");
    fclose(f);
    return err;
  }

  for (IT i = 0; i < coo->nnz; ++i) {
    if (meta->is_symmetric && coo->row[i] < coo->col[i]) continue;
    if (meta->val_type == MM_VAL_TYPE_PATTERN) {
      fprintf(f, "%ld %ld\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1));
    } else if (meta->val_type == MM_VAL_TYPE_REAL) {
      if (meta->val_bytes == 8) {
        fprintf(f, "%ld %ld %.16g\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (double)coo->val[i]);
      } else {
        fprintf(f, "%ld %ld %.8g\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (float)coo->val[i]);
      }
    } else if (meta->val_type == MM_VAL_TYPE_INTEGER) {
      fprintf(f, "%ld %ld %ld\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (long)coo->val[i]);
    } else {
      return MM_UNSUPPORTED_TYPE;
    }
  }

  fclose(f);
  return 0;
}

void mm_set_metadata(Matrix_Metadata* meta, MM_typecode *matcode) {
  if (meta) {
    // Value type
    if (mm_is_real(*matcode)) {
      meta->val_type = MM_VAL_TYPE_REAL;
    } else if (mm_is_integer(*matcode)) {
      meta->val_type = MM_VAL_TYPE_INTEGER;
    } else if (mm_is_pattern(*matcode)) {
      meta->val_type = MM_VAL_TYPE_PATTERN;
    } else {
      fprintf(stderr, "BUG: MM_VAL_TYPE not recognized. Please report this.\n");
      exit(EXIT_FAILURE);
    }
    // Symmetry
    meta->is_symmetric = mm_is_symmetric(*matcode);
  }
}

template<typename IT, typename VT>
Entry<IT, VT>* mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta) {
  if (f == NULL) return NULL;

  int err = mm_read_banner(f, matcode, is_bmtx, meta);
  if (err != 0) {
    fprintf(stderr, "Could not process Matrix Market banner. Error (%d)\n", err);
    return NULL;
  }
  if (mm_is_complex(*matcode)) {
    fprintf(stderr, "Cannot parse complex-valued matrices.\n");
    return NULL;
  }
  if (mm_is_array(*matcode)) {
    fprintf(stderr, "Cannot parse array matrices.\n");
    return NULL;
  }
  if (mm_is_skew(*matcode)) {
    fprintf(stderr, "Cannot parse skew-symmetric matrices.\n");
    return NULL;
  }
  if (mm_is_hermitian(*matcode)) {
    fprintf(stderr, "Cannot parse hermitian matrices.\n");
    return NULL;
  }

  uint64_t _nrows, _ncols, _nnz, mm_nnz;
  if (mm_read_mtx_crd_size(f, &_nrows, &_ncols, &mm_nnz) != 0) {
    fprintf(stderr, "Could not parse matrix size.\n");
    return NULL;
  }

  uint8_t idx_bytes = 0;
  uint8_t val_bytes = 0;
  int IT_required_bytes = required_bytes_index(std::max(_nrows, _ncols));
  
  if (is_bmtx) {
    idx_bytes = mm_get_idx_bytes(*matcode);
    val_bytes = mm_get_val_bytes(*matcode);

    if(!(idx_bytes == 1 || idx_bytes == 2 || idx_bytes == 4 || idx_bytes == 8)
       && !(val_bytes == 1 || val_bytes == 2 || val_bytes == 4 || val_bytes == 8)) {
      fprintf(stderr, "BMTX BUG: this should not happen. idx: %hhu bytes, val: %hhu bytes. Please report this.\n", idx_bytes, val_bytes);
      return NULL;
    }
    if (idx_bytes < IT_required_bytes) {
      fprintf(stderr, "BMTX BUG: this should not happen. Need at least %d bytes, binary is written using %hhu bytes. Please report this.\n", IT_required_bytes, idx_bytes);
      return NULL;
    }
  }
  
  if (sizeof(IT) < (size_t)IT_required_bytes) {
    fprintf(stderr, "Error: Index Type (IT) is too small to represent matrix indices (need at least %d bytes, got %zu bytes).\n", IT_required_bytes, sizeof(IT));
    return NULL;
  }
  

  _nnz = mm_is_symmetric(*matcode) ? mm_nnz * 2 : mm_nnz; // For symmetric matrices THIS IS AN UPPER BOUND
  nrows = static_cast<IT>(_nrows);
  ncols = static_cast<IT>(_ncols);

  Entry<IT, VT> *entries = (Entry<IT, VT> *)malloc(_nnz * sizeof(Entry<IT, VT>));
  err = mm_read_mtx_crd_data<IT, VT>(f, mm_nnz, entries, *matcode, is_bmtx, idx_bytes, val_bytes);
  if (err != 0) {
    printf("Could not parse matrix data (error code: %d).\n", err);
    free(entries);
    fclose(f);
    return NULL;
  }
  fclose(f);

  if (mm_is_symmetric(*matcode)) {
    _nnz = mm_nnz;
    // Duplicate the entries for symmetric matrices
    for (uint64_t i = 0, j = 0; i < mm_nnz; ++i) {
      if (entries[i].row != entries[i].col) { // Do not duplicate diagonal
        entries[j + mm_nnz].row = entries[i].col;
        entries[j + mm_nnz].col = entries[i].row;
        entries[j + mm_nnz].val = entries[i].val;
        ++_nnz;
        ++j;
      }
    }
  }

  nnz = static_cast<IT>(_nnz);
  mm_set_metadata(meta, matcode);

  return entries;
}

// CSR

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  return Distr_MMIO_CSR_local_read_f<IT, VT>(open_file_r(filename), is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
}
// template CSR_local<uint64_t, double>* Distr_MMIO_CSR_local_read(const char *filename, bool expl_val_for_bin_mtx);

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  IT nrows, ncols, nnz;
  MM_typecode matcode;
  Entry<IT, VT> *entries = mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
  if (entries == NULL) return NULL;

  CSR_local<IT, VT> *csr = Distr_MMIO_CSR_local_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
  entries_to_local_csr<IT, VT>(entries, csr);

  free(entries);

  return csr;
}
// template CSR_local<uint64_t, double>* Distr_MMIO_CSR_local_read_f(FILE *f, bool expl_val_for_bin_mtx);

// COO

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  return Distr_MMIO_COO_local_read_f<IT, VT>(open_file_r(filename), is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
}

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  IT nrows, ncols, nnz;
  MM_typecode matcode;
  Entry<IT, VT> *entries = mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
  if (entries == NULL) return NULL;
  
  COO_local<IT, VT> *coo = Distr_MMIO_COO_local_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
  entries_to_local_coo<IT, VT>(entries, coo);
  
  free(entries);

  return coo;
}

template<typename IT, typename VT>
Entry<IT, VT>* Distr_MMIO_COO_local_read_mpi(FILE *f, bool is_bmtx, int comm_size, int myrank, uint32_t *nentries, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  IT nrows, ncols, nnz;
  MM_typecode matcode;

  int err = mm_read_banner(f, &matcode, is_bmtx, meta);
  if (err != 0) {
    fprintf(stderr, "Could not process Matrix Market banner. Error (%d)\n", err);
    return NULL;
  }
  if (mm_is_complex(matcode)) {
    fprintf(stderr, "Cannot parse complex-valued matrices.\n");
    return NULL;
  }
  if (mm_is_array(matcode)) {
    fprintf(stderr, "Cannot parse array matrices.\n");
    return NULL;
  }
  if (mm_is_skew(matcode)) {
    fprintf(stderr, "Cannot parse skew-symmetric matrices.\n");
    return NULL;
  }
  if (mm_is_hermitian(matcode)) {
    fprintf(stderr, "Cannot parse hermitian matrices.\n");
    return NULL;
  }

  uint64_t _nrows, _ncols, _nnz, mm_nnz;
  if (mm_read_mtx_crd_size(f, &_nrows, &_ncols, &mm_nnz) != 0) {
    fprintf(stderr, "Could not parse matrix size.\n");
    return NULL;
  }

  uint8_t idx_bytes = 0;
  uint8_t val_bytes = 0;
  int IT_required_bytes = required_bytes_index(std::max(_nrows, _ncols));

  if (is_bmtx) {
    idx_bytes = mm_get_idx_bytes(matcode);
    val_bytes = mm_get_val_bytes(matcode);

    if(!(idx_bytes == 1 || idx_bytes == 2 || idx_bytes == 4 || idx_bytes == 8)
       && !(val_bytes == 1 || val_bytes == 2 || val_bytes == 4 || val_bytes == 8)) {
      fprintf(stderr, "BMTX BUG: this should not happen. idx: %hhu bytes, val: %hhu bytes. Please report this.\n", idx_bytes, val_bytes);
      return NULL;
    }
    if (idx_bytes < IT_required_bytes) {
      fprintf(stderr, "BMTX BUG: this should not happen. Need at least %d bytes, binary is written using %hhu bytes. Please report this.\n", IT_required_bytes, idx_bytes);
      return NULL;
    }
  }

  if (sizeof(IT) < (size_t)IT_required_bytes) {
    fprintf(stderr, "Error: Index Type (IT) is too small to represent matrix indices (need at least %d bytes, got %zu bytes).\n", IT_required_bytes, sizeof(IT));
    return NULL;
  }


  _nnz = mm_is_symmetric(matcode) ? mm_nnz * 2 : mm_nnz; // For symmetric matrices THIS IS AN UPPER BOUND
  nrows = static_cast<IT>(_nrows);
  ncols = static_cast<IT>(_ncols);

  long int pos = ftell(f);
  uint16_t line_size = (mm_is_pattern(matcode)) ? (2 * idx_bytes) : (2 * idx_bytes + val_bytes) ;
  *nentries = (myrank < _nnz % comm_size) ? (_nnz / comm_size + 1) : (_nnz / comm_size) ;
  uint32_t to_skip = (myrank < _nnz % comm_size) ? (*nentries * myrank) : (*nentries * myrank + _nnz % comm_size ) ;
  Entry<IT, VT> *entries = (Entry<IT, VT> *)malloc(*nentries * sizeof(Entry<IT, VT>));

  if (fseek(f, to_skip*line_size, SEEK_CUR) != 0) {
      perror("fseek failed");
      fclose(f);
      return(NULL);
  }

  long int new_pos = ftell(f);
  fprintf(stdout, "[DEBUG at line %d] process %d: %u nentries, %u to_skip, %ld is the starting position and %ld is the current position (%u B)\n", __LINE__, myrank, *nentries, to_skip, pos, new_pos, to_skip*line_size);

  err = mm_read_mtx_crd_data<IT, VT>(f, *nentries, entries, matcode, is_bmtx, idx_bytes, val_bytes);
  if (err != 0) {
    printf("Could not parse matrix data (error code: %d).\n", err);
    free(entries);
    fclose(f);
    return(NULL);
  }
  fclose(f);

  if (mm_is_symmetric(matcode)) {
    _nnz = mm_nnz;
    // Duplicate the entries for symmetric matrices
    for (uint64_t i = 0, j = 0; i < mm_nnz; ++i) {
      if (entries[i].row != entries[i].col) { // Do not duplicate diagonal
        entries[j + mm_nnz].row = entries[i].col;
        entries[j + mm_nnz].col = entries[i].row;
        entries[j + mm_nnz].val = entries[i].val;
        ++_nnz;
        ++j;
      }
    }
  }

  nnz = static_cast<IT>(_nnz);
  mm_set_metadata(meta, &matcode);

  return entries;
}

template<typename IT, typename VT>
int Distr_MMIO_COO_local_write(COO_local<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta) {
  return Distr_MMIO_COO_local_write_f(coo, open_file_w(filename), write_as_binary, meta);
}

template<typename IT, typename VT>
int Distr_MMIO_COO_local_write_f(COO_local<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta) {
  if (meta->mm_header.empty()) {
    meta->mm_header = "%%MatrixMarket matrix coordinate";

    switch (meta->val_type) {
      case MM_VAL_TYPE_REAL:    { meta->mm_header += std::string(MM_REAL_STR);    break; }
      case MM_VAL_TYPE_INTEGER: { meta->mm_header += std::string(MM_INT_STR);     break; }
      case MM_VAL_TYPE_PATTERN: { meta->mm_header += std::string(MM_PATTERN_STR); break; }
      default:                  { fprintf(stderr, "BUG: MM_VAL_TYPE not recognized\n"); return 100; }
    }

    meta->mm_header += meta->is_symmetric ? "symmetric" : "general";
  }
  
  return write_as_binary ? write_binary_matrix_market(f, coo, meta) : write_matrix_market(f, coo, meta);
}


MMIO_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, double)
MMIO_EXPLICIT_TEMPLATE_INST(int, float)
MMIO_EXPLICIT_TEMPLATE_INST(int, double)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)

// ---------------------------------------------------------------------------------------------------------------------------------
//                                          Probably to move in a dedicated file
// ---------------------------------------------------------------------------------------------------------------------------------

ProcessGrid * make_process_grid(int row_size, int col_size, int node_size)
{
    ASSERT((row_size>0), "row_size must be > 0");
    ASSERT((col_size>0), "col_size must be > 0");
    ASSERT((node_size>0), "node_size must be > 0");

    ASSERT((row_size==col_size), "Currently only square grids are supported");

    ProcessGrid * grid = (ProcessGrid*)malloc(sizeof(ProcessGrid));

    grid->rsz = row_size;
    grid->csz = col_size;
    grid->nsz = node_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &(grid->grk));
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->gsz));
    grid->world_comm = MPI_COMM_WORLD;

    ASSERT((grid->gsz % (row_size*col_size) == 0), "Total number of processes must be a multiple of row_size * col_size");

    // Compute 3D coordinates from linear rank
    int i = grid->grk / (col_size * node_size);      // row index
    int remainder = grid->grk % (col_size * node_size);
    int j = remainder / node_size;                   // col index
    int k = remainder % node_size;                   // node index

    // --- Node communicator (same i,j, different k)
    int node_color = i * col_size + j; // unique per (i,j)
    int node_key   = k;
    MPI_Comm_split(MPI_COMM_WORLD, node_color, node_key, &(grid->node_comm));

    // --- Row communicator: (i, *, k) → fix i and k, vary j
    int row_color = i * node_size + k; // unique per (i,k)
    int row_key   = j;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &(grid->row_comm));

    // --- Column communicator: (*, j, k) → fix j and k, vary i
    int col_color = j * node_size + k; // unique per (j,k)
    int col_key   = i;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &(grid->col_comm));

    MPI_Barrier(grid->world_comm);

    MPI_Comm_rank(grid->row_comm, &(grid->rrk));
    MPI_Comm_size(grid->row_comm, &(grid->rsz));

    MPI_Comm_rank(grid->col_comm, &(grid->crk));
    MPI_Comm_size(grid->col_comm, &(grid->csz));

    MPI_Comm_rank(grid->node_comm, &(grid->nrk));
    MPI_Comm_size(grid->node_comm, &(grid->nsz));

//     if (grid->rsz != row_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
//     if (grid->csz != col_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
//     if (grid->nsz != node_size) MPI_Abort(MPI_COMM_WORLD, __LINE__);

    return grid;
}

void print_process_grid(const ProcessGrid *grid, FILE* fp)
{
	if (grid->grk==0)
    {
        fprintf(fp, "========================\n");
        fprintf(fp, " ProcessGrid Details \n");
        fprintf(fp, "========================\n");
        fprintf(fp, "Total processes:\t %d\n", grid->gsz);
        fprintf(fp, "row size:\t %d\n", grid->rsz);
        fprintf(fp, "col size:\t %d\n", grid->csz);
        fprintf(fp, "node size:\t %d\n", grid->nsz);

    }
    MPI_Barrier(MPI_COMM_WORLD);
	sleep(1);

    for (int i=0; i<grid->gsz; i++)
    {
        if (grid->grk== i)
        {
            fprintf(fp, "----- Process %d -----\n", grid->grk);
            fprintf(fp, "Rank:\t %d\n", grid->grk);
            fprintf(fp, "row rank:\t %d\n", grid->rrk);
            fprintf(fp, "col rank:\t %d\n", grid->crk);
            fprintf(fp, "node rank:\t %d\n", grid->nrk);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    sleep(1);
    if (grid->grk == 0) fprintf(fp, "========================\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

// ==================================================== Partitioning functions ==================================================
/*  Functions and macros to compute the partitioning
 *
 *  The partitioning could be hyerarcycal ...to explain...
 *
 */

void init_mypart(Partitioning *self) {
    self->my_part_str = NULL;

    self->proc_grid = NULL;

    self->operand_type = 'l';

    self->global_matrix_rows = 0;
    self->global_matrix_cols = 0;
    self->group_matrix_rows = 0;
    self->group_matrix_cols = 0;
    self->local_matrix_rows = 0;
    self->local_matrix_cols = 0;

    self->edge2group = NULL;
    self->edge2nodeprocess = NULL;
    self->globalcol2groupcol = NULL;
    self->globalrow2grouprow = NULL;
    self->groupcol2localcol = NULL;
    self->grouprow2localrow = NULL;
}

void set_mypart_str (Partitioning *self) {
	switch (self->my_part_type) {
		case PART_RSLICING:
			self->my_part_str = "partRslicing";
			break;
		case PART_BCYCLE:
			self->my_part_str = "partBcycle";
			break;
        case PART_NAIVE_CYCLE:
            self->my_part_str = "partNaiveCycle";
            break;
        case PART_BCYCLE_CYCLE:
            self->my_part_str = "partBcycleCycle";
            break;
        case PART_RSLICING_CYCLE:
			self->my_part_str = "partRslicingCycle";
			break;
		default:
			self->my_part_str = "partNaive";
			break;
    }
}

void set_mypart_transpose(Partitioning *self, char operand_type) {
//     ASSERT((operand_type=='l')||(operand_type=='r'), "Error in function %s, operand_type must be 'l' (left) or 'r' (right).", __func__);
    self->operand_type = (operand_type=='l') ? 'l' : 'r' ;
}

void set_mypart_type (Partitioning *self, PartitioningType partype, char operand_type) {
    init_mypart(self);
	self->my_part_type = partype;
	set_mypart_str(self);
    set_mypart_transpose(self, operand_type);
}

void set_mypart_grid (Partitioning *self, ProcessGrid* proc_grid) {
	self->proc_grid = proc_grid;
}

void set_mypart_globaldim (Partitioning *self, uint64_t n, uint64_t m) {
	self->global_matrix_rows = n;
	self->global_matrix_cols = m;
}

// Common macros: these macros are commonly used by all the partition functions

/*
 * This macro compute the partitioning chunk size. GS stands for global size while NC stands for number of chunks
 * To menage global sizes that are not multiples of the number of chunks, we define the chunk size as ((GS)/(NC))+1;
 *   this mean that the last chunk can have less elements than the others (however, the unbalance is less then NC-1).
 */
#define PART_CHUNK_SIZE(GS, NC) (((GS)%(NC))==0) ? ((GS)/(NC)) : (((GS)/(NC))+1)


/*
 * This macro retrive the process id from them grid coordinates. The M parameteer is neaded to retrive the grid and
 * represents the number of columns inside the grid. RI and CI stands for row id and column id and represent the
 * process grid coordinates.
 *
 * NOTE: currently unused
 */
#define GRIDCOO2PROCID(RI, CI, M) ((RI)*(M) + (CI))

/* Edge to process functions
 *
 * These functions retrive the owner process starting by them 'global' indices.
 * Function are 1d or 2d functions; these two classes differs by the nprocess input parameteers. For
 * each of those, a typedef is specified.
 *
 * Currently, the only 1D scheme supported is the naive one.
 * Currently, the 2D supported scheme are naive, row-slicing and block-cycle.
 *
 */

// Defined in the header
// typedef uint64_t (*Edge2proc1dFunction)(uint64_t i, uint64_t j,
// 										uint64_t nrows, uint64_t ncols,
// 										int nprocs);

uint64_t edge2proc_1d_naive(uint64_t idx,
                       uint64_t dimtosplit,
                       int nprocs)
{
	int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs);
    return (idx / chunk_size );
}

uint64_t edge2proc_1d_cycle(uint64_t idx,
                       uint64_t dimtosplit,
                       int nprocs)
{
	int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs*nprocs);
    return ( (idx/chunk_size) % nprocs );
}

// Defined in the header
// typedef uint64_t (*Edge2proc2dFunction)(uint64_t i, uint64_t j,
// 										uint64_t nrows, uint64_t ncols,
// 										int procs_per_row, int procs_per_col);

uint64_t edge2proc_2d_naive(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
	int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
	int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);
    return ( (i/rows_chunk_size)*procs_per_row + j/cols_chunk_size );
}

uint64_t edge2proc_2d_rowslicing(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);

    int process_col_coord = j/cols_chunk_size;
    int process_row_coord = (i/rows_chunk_size) % procs_per_col;

	return ( process_row_coord*procs_per_row + process_col_coord);
}

uint64_t edge2proc_2d_rowslicing_transpose(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

    int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    int process_row_coord = i/rows_chunk_size;
	return ( process_row_coord*procs_per_row + process_col_coord);
}

uint64_t edge2proc_2d_blockcycle(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
	int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
	int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

	int process_row_coord = (i/rows_chunk_size) % procs_per_col;
	int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    return ( process_row_coord*procs_per_row + process_col_coord);
}


uint64_t edge2proc_2d_1d(uint64_t i, uint64_t j,
                           uint64_t m, uint64_t n,
                           ProcessGrid * proc_grid, int transpose)
{
    ASSERT((n % proc_grid->csz == 0), "pc %d does not divide n %lu\n", proc_grid->csz, n); // Process grid must evenly divide matrix dims
    ASSERT((m % (proc_grid->rsz * proc_grid->nsz) == 0), "pr*pz %d does not divide m %lu\n", proc_grid->nsz * proc_grid->nsz, m);

	int rpg = m / proc_grid->rsz;
    int cpg = n / proc_grid->csz;


    int gid = (i / rpg) * proc_grid->csz + j / cpg;

    int rpp = rpg / proc_grid->nsz;
    int intragroup_id = (i % rpg) / rpp;

    int pid = gid * proc_grid->nsz + intragroup_id;

#if DEBUG_PARTITION
    printf("(%lu, %lu) mapped to %d. rpg:%d, cpg:%d, gid:%d, rpp:%d, ig_id: %d\n",
                    i, j, pid, rpg, cpg, gid, rpp, intragroup_id);
    FLUSH_WAIT(0.5);
#endif

    ASSERT((pid < proc_grid->gsz), "pid is %d, must be < %d\n", pid, proc_grid->gsz);

    return pid;
}


/* Global indices to local indices
 *
 * These functions retrive the lacal indices starting by the global indices. Note that global and local
 * are independent by the 3D process grid. All these function are base partitioning that can be composed
 * togheter to generate hybrid partitionings (this is also true for the edge to process functions).
 *
 * Function are 1d or 2d functions; these two classes differs by the nprocess input parameteers. For
 * each of those, a typedef is specified.
 *
 * Currently, the only 1D scheme supported is the naive one.
 * Currently, the 2D supported scheme are naive, row-slicing and block-cycle.
 *
 */

// General function for translating global indices to local indices (into header)
// typedef uint64_t (*GlobalIdx2LocalIdx)(uint64_t globalid, uint64_t globalsize, int nprocs);

uint64_t globalindex2localindex_naive(uint64_t globalid, uint64_t globalsize, int nprocs)
{
	int chunk_size = PART_CHUNK_SIZE(globalsize,nprocs);
    return (globalid % chunk_size );
}

uint64_t globalindex2localindex_cycle(uint64_t globalid, uint64_t globalsize, int nprocs)
{
    int chunk_size = PART_CHUNK_SIZE(globalsize,nprocs*nprocs);
	return ( ((globalid/chunk_size)/nprocs)*chunk_size + (globalid%chunk_size) );
}

/* Each partitioning scheme use one of the previous defined general function to
 *  translate global indices to local indices.
 *
 * Naive partitioning:
 *     use naive function both for rows and columns.
 *
 * Row slicing partitioning:
 *     use cycle function for rows and naive function for columns.
 *
 * Block cycle partitioning:
 *     use cycle function both for rows and columns.
 *
 */

// Others

void set_mypart_groupdim (Partitioning *self) {
	ASSERT((self->proc_grid!=NULL), "%s call before self->proc_grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

	self->group_matrix_rows = PART_CHUNK_SIZE(self->global_matrix_rows, (self->proc_grid)->rsz);
	self->group_matrix_cols = PART_CHUNK_SIZE(self->global_matrix_cols, (self->proc_grid)->csz);
}

void set_mypart_localdim (Partitioning *self) {
	ASSERT((self->proc_grid!=NULL), "%s call before self->proc_grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

    ASSERT(((self->group_matrix_rows!=0)&&(self->group_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tgroup_matrix_rows: %lu\n\tgroup_matrix_cols: %lu\n", \
		__func__, self->group_matrix_rows, self->group_matrix_cols);

    // BUG, tmp fix
//     if (self->operand_type == 'l') {
//         self->local_matrix_rows = PART_CHUNK_SIZE(self->group_matrix_rows, (self->proc_grid)->pz);
//         self->local_matrix_cols = self->group_matrix_cols;
//     } else {
//         self->local_matrix_rows = self->group_matrix_rows;
//         self->local_matrix_cols = PART_CHUNK_SIZE(self->group_matrix_cols, (self->proc_grid)->pz);
//     }
    self->local_matrix_rows = PART_CHUNK_SIZE(self->group_matrix_rows, (self->proc_grid)->nsz);
    self->local_matrix_cols = self->group_matrix_cols;
}

void set_mypart_functions (Partitioning *self) {
	ASSERT((self->proc_grid!=NULL), "%s call before self->proc_grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

	ASSERT(((self->group_matrix_rows!=0)&&(self->group_matrix_cols!=0)), \
		"%s call before group matrix dim are set:\n\tgroup_matrix_rows: %lu\n\tgroup_matrix_cols: %lu\n", \
		__func__, self->group_matrix_rows, self->group_matrix_cols);


	switch (self->my_part_type) {
		case PART_BCYCLE: {
			// LOAD RSLICING PARTITIONING
			self->edge2group = edge2proc_2d_blockcycle;
			self->edge2nodeprocess = edge2proc_1d_naive;
			self->globalcol2groupcol = globalindex2localindex_cycle;
			self->globalrow2grouprow = globalindex2localindex_cycle;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case PART_BCYCLE_CYCLE: {
			// LOAD RSLICING PARTITIONING
			self->edge2group = edge2proc_2d_blockcycle;
			self->edge2nodeprocess = edge2proc_1d_cycle;
			self->globalcol2groupcol = globalindex2localindex_cycle;
			self->globalrow2grouprow = globalindex2localindex_cycle;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_cycle;
			break;
		}

		case PART_RSLICING: {
			// PART_RSLICING and PART_NAIVE use the same column partitioning
			self->edge2nodeprocess = edge2proc_1d_naive;

            if (self->operand_type == 'l') {
                self->edge2group = edge2proc_2d_rowslicing;
                self->globalcol2groupcol = globalindex2localindex_naive;
                self->globalrow2grouprow = globalindex2localindex_cycle;
            } else {
                self->edge2group = edge2proc_2d_rowslicing_transpose;
                self->globalcol2groupcol = globalindex2localindex_cycle;
                self->globalrow2grouprow = globalindex2localindex_naive;
            }

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case PART_RSLICING_CYCLE: {
			// PART_RSLICING and PART_NAIVE use the same column partitioning
			self->edge2nodeprocess = edge2proc_1d_cycle;

            if (self->operand_type == 'l') {
                self->edge2group = edge2proc_2d_rowslicing;
                self->globalcol2groupcol = globalindex2localindex_naive;
                self->globalrow2grouprow = globalindex2localindex_cycle;
            } else {
                self->edge2group = edge2proc_2d_rowslicing_transpose;
                self->globalcol2groupcol = globalindex2localindex_cycle;
                self->globalrow2grouprow = globalindex2localindex_naive;
            }

			self->groupcol2localcol = (self->operand_type=='l') ? globalindex2localindex_naive : globalindex2localindex_cycle ;
			self->grouprow2localrow = (self->operand_type=='l') ? globalindex2localindex_cycle : globalindex2localindex_naive ;
			break;
		}

		case PART_NAIVE: {
			// PART_RSLICING and PART_NAIVE use the same column partitioning
			self->edge2group = edge2proc_2d_naive;
			self->edge2nodeprocess = edge2proc_1d_naive;
			self->globalcol2groupcol = globalindex2localindex_naive;
			self->globalrow2grouprow = globalindex2localindex_naive;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case PART_NAIVE_CYCLE: {
			// PART_RSLICING and PART_NAIVE use the same column partitioning
			self->edge2group = edge2proc_2d_naive;
			self->edge2nodeprocess = edge2proc_1d_cycle;
			self->globalcol2groupcol = globalindex2localindex_naive;
			self->globalrow2grouprow = globalindex2localindex_naive;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_cycle;
			break;
		}
	}
}

void set_mypart (Partitioning *self, PartitioningType partype, ProcessGrid* proc_grid, uint64_t glob_nrows, uint64_t glob_ncols) {
    set_mypart_type(self, partype);
    set_mypart_grid(self, proc_grid);
    set_mypart_globaldim(self, glob_nrows, glob_ncols);

    set_mypart_groupdim(self);
    set_mypart_functions(self);
}

// Functions from edge to process
uint64_t edge2group (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->edge2group(glob_row_id, glob_col_id, self->global_matrix_rows, self->global_matrix_cols, (self->proc_grid)->csz, (self->proc_grid)->rsz) );
}

uint64_t edge2nodeprocess (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     uint64_t idxtosplit = (self->operand_type=='l') ? (globalrow2grouprow(self, glob_row_id)) : (globalcol2groupcol(self, glob_col_id)) ;
//     uint64_t dimtosplit = (self->operand_type=='l') ? self->group_matrix_rows : self->group_matrix_cols ;
    // BUG, tmp fix
    uint64_t idxtosplit = (globalrow2grouprow(self, glob_row_id));
    uint64_t dimtosplit = self->group_matrix_rows;
	return( self->edge2nodeprocess(idxtosplit, dimtosplit, (self->proc_grid)->nsz) );
}

// Function from upper index to inner index
uint64_t globalcol2groupcol (Partitioning *self, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->globalcol2groupcol(glob_col_id, self->global_matrix_cols, (self->proc_grid)->csz) );
}

uint64_t globalrow2grouprow (Partitioning *self, uint64_t glob_row_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->globalrow2grouprow(glob_row_id, self->global_matrix_rows, (self->proc_grid)->rsz) );
}

uint64_t groupcol2localcol (Partitioning *self, uint64_t grp_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     int ncolsplit = (self->operand_type=='l') ? 1 : ((self->proc_grid)->pz) ;
    int ncolsplit = 1; // BUG, tmp fix
	return( self->groupcol2localcol(grp_col_id, self->group_matrix_cols, ncolsplit) );
}
uint64_t grouprow2localrow (Partitioning *self, uint64_t grp_row_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     int nrowsplit = (self->operand_type=='l') ? ((self->proc_grid)->pz) : 1 ;
    int nrowsplit = ((self->proc_grid)->nsz); // BUG, tmp fix
	return( self->grouprow2localrow(grp_row_id, self->group_matrix_rows, nrowsplit) );
}

// Composed function: from global index to local index and from edge to global rank
uint64_t globalcol2localcol (Partitioning *self, uint64_t glob_col_id) {
	uint64_t grp_col_id = globalcol2groupcol (self, glob_col_id);
	return(  groupcol2localcol(self, grp_col_id) );
}
uint64_t globalrow2localrow (Partitioning *self, uint64_t glob_row_id) {
	uint64_t grp_row_id = globalrow2grouprow (self, glob_row_id);
	return( grouprow2localrow(self, grp_row_id) );
}

uint64_t edge2globalprocess (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
	int gid = edge2group(self, glob_row_id, glob_col_id); // BUG for the row slicing we need to solve the problem of the transpose
	int intragroup_id = edge2nodeprocess(self, glob_row_id, glob_col_id);
	int pid = gid * ((self->proc_grid)->nsz) + intragroup_id;

#if DEBUG_PARTITION
    printf("%s: (%lu, %lu) mapped to %d. gid: %d, ig_id: %d\n",
                    self->my_part_str, glob_row_id, glob_col_id, pid, gid, intragroup_id);
    FLUSH_WAIT(0.5);
#endif
    return(pid);
}

#include <algorithm>
#include <vector>

// Sort entries by owner, producing a new sorted vector
template<typename IT, typename VT>
Entry<IT, VT>* sortEntriesByOwner(const Entry<IT, VT>* entries, const int* owner, size_t nentries) {
    // Combine entries and owner into a vector of pairs
    std::vector<std::pair<int, Entry<IT, VT>>> combined(nentries);
    for (size_t i = 0; i < nentries; ++i) {
        combined[i] = { owner[i], entries[i] };
    }

    // Sort by owner
    std::sort(combined.begin(), combined.end(),
              [](const std::pair<int, Entry<IT, VT>>& a, const std::pair<int, Entry<IT, VT>>& b) {
                  return a.first < b.first;
              });

    // Allocate new array for sorted entries
    Entry<IT, VT>* sorted_entries = (Entry<IT, VT>*)malloc(nentries * sizeof(Entry<IT, VT>));
    for (size_t i = 0; i < nentries; ++i) {
        sorted_entries[i] = combined[i].second;
    }

    return sorted_entries;
}

