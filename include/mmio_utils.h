#ifndef MM_IO_UTILS_H
#define MM_IO_UTILS_H

template<typename IT, typename VT>
void print_csr(CSR_local<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void print_csr_as_dense(CSR_local<IT, VT> *csr, std::string header="", FILE* fp=stdout);

template<typename IT, typename VT>
void print_coo(COO_local<IT, VT> *coo, std::string header="", FILE* fp=stdout);

#endif
