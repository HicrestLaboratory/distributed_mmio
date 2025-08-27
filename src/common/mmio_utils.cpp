#include <stdio.h>
#include <string>
#include <vector>

#include "../../include/common/mmio.h"
#include "../../include/common/mmio_utils.h"

#define MMIO_UTILS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template void print_csr(CSR<IT, VT> *csr, std::string header, FILE* fp); \
  template void print_csr_as_dense<IT, VT>(CSR<IT, VT> *csr, std::string header, FILE* fp); \
  template void print_coo(COO<IT, VT> *coo, std::string header, FILE* fp); \

template<typename IT, typename VT>
void print_csr(CSR<IT, VT> *csr, std::string header, FILE* fp) {
  if (header != "") {
    fprintf(fp, "%s -- ", header.c_str());
  }
  std::string I_FMT = "%3u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%3lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  fprintf(fp, fmt, csr->nrows, csr->ncols, csr->nnz);

  snprintf(fmt, 100, "%s ", I_FMT.c_str());
  fprintf(fp, "idx   : ");
  for (IT i = 0; i < csr->nnz; ++i) fprintf(fp, fmt, i);
  fprintf(fp, "\nrowptr: ");
  for (IT i = 0; i <= csr->nrows; ++i) fprintf(fp, fmt, csr->row_ptr[i]);
  fprintf(fp, "\ncolidx: ");
  for (IT i = 0; i < csr->nnz; ++i) fprintf(fp, fmt, csr->col_idx[i]);
  if (csr->val != NULL) {
    fprintf(fp, "\nval:    ");
    for (IT i = 0; i < csr->nnz; ++i) {
      fprintf(fp, "%.1f ", csr->val[i]); // TODO handle different VT
    }
  }
  fprintf(fp, "\n");
}

template<typename IT, typename VT>
void print_csr_as_dense(CSR<IT, VT> *csr, std::string header, FILE* fp) {
  std::vector<std::vector<VT>> dense_matrix(csr->nrows, std::vector<VT>(csr->ncols, 0.0f));
  for (IT row = 0; row < csr->nrows; ++row) {
    for (IT idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; ++idx) {
      IT col = csr->col_idx[idx];
      dense_matrix[row][col] = csr->val != NULL ? csr->val[idx] : 1.0f; // TODO handle different VT
    }
  }
  if (header != "") {
      fprintf(fp, "%s -- ", header.c_str());
  }

  std::string I_FMT = "%3u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%3lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  fprintf(fp, fmt, csr->nrows, csr->ncols, csr->nnz);
  
  for (IT row = 0; row < csr->nrows; ++row) {
    for (IT col = 0; col < csr->ncols; ++col) {
      if (dense_matrix[row][col] == 0)
        fprintf(fp, "   - ");
      else
        fprintf(fp, "%4.0f ", dense_matrix[row][col]); // TODO handle different VT
    }
    fprintf(fp, "\n");
  }
}

template<typename IT, typename VT>
void print_coo(COO<IT, VT> *coo, std::string header, FILE* fp) {
  if (header != "") {
    fprintf(fp, "%s -- ", header.c_str());
  }
  
  std::string I_FMT = "%4u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%4lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  fprintf(fp, fmt, coo->nrows, coo->ncols, coo->nnz);

  snprintf(fmt, 100, "%s ", I_FMT.c_str());
  fprintf(fp, "idx: ");
  for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, i);
  fprintf(fp, "\nrow: ");
  for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, coo->row[i]);
  fprintf(fp, "\ncol: ");
  for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, coo->col[i]);
  if (coo->val != NULL) {
    fprintf(fp, "\nval: ");
    for (IT i = 0; i < coo->nnz; ++i) {
      fprintf(fp, "%4.1f ", coo->val[i]); // TODO handle different VT
    }
  }
  fprintf(fp, "\n");
}


MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
