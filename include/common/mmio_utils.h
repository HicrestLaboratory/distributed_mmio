#ifndef __DMMIO_UTILS_H__
#define __DMMIO_UTILS_H__

template<typename IT, typename VT>
void DMMIO_print_CSR(CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void DMMIO_print_CSR_as_dense(CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void DMMIO_print_COO(COO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void DMMIO_print_COO_as_dense(COO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

#endif
