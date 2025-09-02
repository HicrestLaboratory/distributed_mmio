#include <ccutils/macros.h>

#include "../../include/dmmio/partitioning.h"

namespace dmmio::partitioning {

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

  uint64_t edge2proc_1d_naive(
    uint64_t idx,
    uint64_t dimtosplit,
    int nprocs
  ) {
    int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs);
    return (idx / chunk_size );
  }

  uint64_t edge2proc_1d_cycle(
    uint64_t idx,
    uint64_t dimtosplit,
    int nprocs
  ) {
    int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs*nprocs);
    return ( (idx/chunk_size) % nprocs );
  }

  // Defined in the header
  // typedef uint64_t (*Edge2proc2dFunction)(uint64_t i, uint64_t j,
  // 										uint64_t nrows, uint64_t ncols,
  // 										int procs_per_row, int procs_per_col);

  uint64_t edge2proc_2d_naive(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  ) {
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);
    return ( (i/rows_chunk_size)*procs_per_row + j/cols_chunk_size );
  }

  uint64_t edge2proc_2d_rowslicing(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  ) {
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);

    int process_col_coord = j/cols_chunk_size;
    int process_row_coord = (i/rows_chunk_size) % procs_per_col;

    return process_row_coord*procs_per_row + process_col_coord;
  }

  uint64_t edge2proc_2d_rowslicing_transpose(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  ) {
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

    int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    int process_row_coord = i/rows_chunk_size;
    return ( process_row_coord*procs_per_row + process_col_coord);
  }

  uint64_t edge2proc_2d_blockcycle(
    uint64_t i, uint64_t j,
    uint64_t nrows, uint64_t ncols,
    int procs_per_row, int procs_per_col
  ) {
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

    int process_row_coord = (i/rows_chunk_size) % procs_per_col;
    int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    return ( process_row_coord*procs_per_row + process_col_coord);
  }

  uint64_t edge2proc_2d_1d(
    uint64_t i, uint64_t j,
    uint64_t m, uint64_t n,
    ProcessGrid * grid, int transpose
  ) {
    ASSERT((n % grid->col_size == 0), "pc %d does not divide n %lu\n", grid->col_size, n); // Process grid must evenly divide matrix dims
    ASSERT((m % (grid->row_size * grid->node_size) == 0), "pr*pz %d does not divide m %lu\n", grid->node_size * grid->node_size, m);

    int rpg = m / grid->row_size;
    int cpg = n / grid->col_size;


    int gid = (i / rpg) * grid->col_size + j / cpg;

    int rpp = rpg / grid->node_size;
    int intragroup_id = (i % rpg) / rpp;

    int pid = gid * grid->node_size + intragroup_id;

  #if DEBUG_PARTITION
    printf("(%lu, %lu) mapped to %d. rpg:%d, cpg:%d, gid:%d, rpp:%d, ig_id: %d\n",
                    i, j, pid, rpg, cpg, gid, rpp, intragroup_id);
    FLUSH_WAIT(0.5);
  #endif

    ASSERT((pid < grid->global_size), "pid is %d, must be < %d\n", pid, grid->global_size);
    
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

  uint64_t globalindex2localindex_naive(uint64_t globalid, uint64_t globalsize, int nprocs) {
    int chunk_size = PART_CHUNK_SIZE(globalsize,nprocs);
    return (globalid % chunk_size );
  }

  uint64_t globalindex2localindex_cycle(uint64_t globalid, uint64_t globalsize, int nprocs) {
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

} // namespace dmmio::partitioning