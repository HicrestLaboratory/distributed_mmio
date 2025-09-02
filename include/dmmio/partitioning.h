#ifndef DMMIO_PARTITIONING_H
#define DMMIO_PARTITIONING_H

#include "dmmio.h"

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


namespace dmmio::partitioning {

  // Internal edge-to-proc mapping
  uint64_t edge2proc_2d_1d(
    uint64_t i, uint64_t j,
    uint64_t m, uint64_t n,
    ProcessGrid* grid,
    int transpose
  );

  // Accessors
  uint64_t edge2group(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id);
  uint64_t edge2node(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id);

  uint64_t globalcol2groupcol(Partitioning* self, uint64_t glob_col_id);
  uint64_t globalrow2grouprow(Partitioning* self, uint64_t glob_row_id);
  uint64_t groupcol2localcol(Partitioning* self, uint64_t grp_col_id);
  uint64_t grouprow2localrow(Partitioning* self, uint64_t grp_row_id);

  uint64_t globalcol2localcol(Partitioning* self, uint64_t glob_col_id);
  uint64_t globalrow2localrow(Partitioning* self, uint64_t glob_row_id);

  uint64_t edge2globalprocess(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id);

  uint64_t edge2proc_1d_naive(
    uint64_t idx,
    uint64_t dimtosplit,
    int nprocs
  );

  uint64_t edge2proc_1d_cycle(
    uint64_t idx,
    uint64_t dimtosplit,
    int nprocs
  );

  uint64_t edge2proc_2d_naive(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  );

  uint64_t edge2proc_2d_rowslicing(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  );

  uint64_t edge2proc_2d_rowslicing_transpose(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  );

  uint64_t edge2proc_2d_blockcycle(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  );

  uint64_t edge2proc_2d_1d(
    uint64_t i, uint64_t j,
    uint64_t m, uint64_t n,
    ProcessGrid * grid, int transpose
  );

  uint64_t globalindex2localindex_naive(uint64_t globalid, uint64_t globalsize, int nprocs);

  uint64_t globalindex2localindex_cycle(uint64_t globalid, uint64_t globalsize, int nprocs);

} // namespace dmmio::partitioning

#endif // DMMIO_PARTITIONING_H
