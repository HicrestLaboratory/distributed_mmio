#ifndef __DMMIO_UTILS_H__
#define __DMMIO_UTILS_H__

#include "dmmio.h"

namespace dmmio::utils {

    void ProcessGrid_print(const ProcessGrid* grid, FILE* fp = stdout);

} // namespace dmmio::utils

#endif // __DMMIO_UTILS_H__