#ifndef __DMMIO_UTILS_H__
#define __DMMIO_UTILS_H__

#include "dmmio.h"
#include <string>

namespace dmmio::utils {

    void ProcessGrid_print(const ProcessGrid* grid, FILE* fp = stdout);
    int ProcessGrid_graph(const dmmio::ProcessGrid *grid, FILE* fp, bool host_gpu_id_print = false);

    template<typename IT, typename VT>
    void DCOO_print_as_dense(dmmio::DCOO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

} // namespace dmmio::utils

#endif // __DMMIO_UTILS_H__
