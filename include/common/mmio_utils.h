#ifndef __DMMIO_UTILS_H__
#define __DMMIO_UTILS_H__

template<typename IT, typename VT>
void print_csr(CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void print_csr_as_dense(CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void print_coo(COO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

#endif
