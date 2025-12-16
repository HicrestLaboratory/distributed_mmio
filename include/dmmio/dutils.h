#ifndef __DMMIO_UTILS_H__
#define __DMMIO_UTILS_H__

#include "dmmio.h"

namespace dmmio::utils {

    void ProcessGrid_print(const ProcessGrid* grid, FILE* fp = stdout);
    int ProcessGrid_graph(const dmmio::ProcessGrid *grid, FILE* fp, bool host_gpu_id_print = false);

} // namespace dmmio::utils

#endif // __DMMIO_UTILS_H__
