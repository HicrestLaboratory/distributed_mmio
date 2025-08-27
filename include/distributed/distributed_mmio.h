/* 
*   Matrix Market I/O library for ANSI C
*   See http://math.nist.gov/MatrixMarket for details.
*/

#ifndef __DMMIO_DISTRIBUTED_H__
#define __DMMIO_DISTRIBUTED_H__

#include <stdint.h>
#include <stdio.h>
#include <string>
#include <mpi.h>

#include "../../include/common/mmio.h"

struct ProcessGrid {
    int grk; // global rank
    int gsz; // size of COMM_WORLD

    int rsz, csz, nsz; // row, column and node communicator sizes
    int rrk, crk, nrk; // row, column and node communicator ranks

    MPI_Comm world_comm, row_comm, col_comm, node_comm;
} typedef ProcessGrid;

ProcessGrid * make_process_grid(int row_size, int col_size, int node_size);
void print_process_grid(const ProcessGrid *grid, FILE* fp = stdout);

// ===================================================== Partitioning functions ===================================================
/* Partitioning functions to be used when reading in matrices */

typedef enum {
    PART_NAIVE,         // 0
    PART_BCYCLE,        // 1
    PART_RSLICING,      // 2
    PART_NAIVE_CYCLE,   // 3
    PART_BCYCLE_CYCLE,  // 4
    PART_RSLICING_CYCLE // 5
} PartitioningType;

typedef uint64_t (*Edge2proc1dFunction)(uint64_t idx,
										uint64_t dimtosplit,
										int nprocs);

typedef uint64_t (*Edge2proc2dFunction)(uint64_t i, uint64_t j,
										uint64_t nrows, uint64_t ncols,
										int procs_per_row, int procs_per_col);

typedef uint64_t (*GlobalIdx2LocalIdx)(uint64_t globalid, uint64_t globalsize, int nprocs);

uint64_t edge2proc_2d_1d(uint64_t i, uint64_t j,
                         uint64_t m, uint64_t n,
                         ProcessGrid * proc_grid, int transpose);

// Partitioning structure definition
typedef struct {
    PartitioningType my_part_type;
    const char *my_part_str;
    char operand_type;

    ProcessGrid * proc_grid;

    uint64_t global_matrix_rows;
    uint64_t global_matrix_cols;
    uint64_t group_matrix_rows;
    uint64_t group_matrix_cols;
    uint64_t local_matrix_rows;
    uint64_t local_matrix_cols;

    Edge2proc2dFunction edge2group;
    Edge2proc1dFunction edge2nodeprocess;
    GlobalIdx2LocalIdx  globalcol2groupcol;
    GlobalIdx2LocalIdx  globalrow2grouprow;
    GlobalIdx2LocalIdx  groupcol2localcol;
    GlobalIdx2LocalIdx  grouprow2localrow;
} Partitioning;

// Moved from above
void compute_local_matrix_dims(const Partitioning* part,
                               const uint64_t global_nrows,
                               const uint64_t global_ncols,
                               uint64_t * loc_nrows,
                               uint64_t * loc_ncols);

// mypart global variable set-up
// extern Partitioning mypart;
void set_mypart_type (Partitioning *self, PartitioningType partype, char operand_type = 'l');
void set_mypart_grid (Partitioning *self, ProcessGrid* proc_grid);
void set_mypart_globaldim (Partitioning *self, uint64_t n, uint64_t m);
void set_mypart_groupdim (Partitioning *self);
void set_mypart_localdim (Partitioning *self);
void set_mypart_functions (Partitioning *self);

void set_mypart (Partitioning *self, PartitioningType partype, ProcessGrid* proc_grid, uint64_t glob_nrows, uint64_t glob_ncols);

// Functions from edge to process
uint64_t edge2group (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);
uint64_t edge2nodeprocess (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);

// Function from upper index to inner index
uint64_t globalcol2groupcol (Partitioning *self, uint64_t glob_col_id);
uint64_t globalrow2grouprow (Partitioning *self, uint64_t glob_row_id);
uint64_t groupcol2localcol  (Partitioning *self, uint64_t grp_col_id);
uint64_t grouprow2localrow  (Partitioning *self, uint64_t grp_row_id);

// Composed function: from global index to local index and from edge to global rank
uint64_t globalcol2localcol (Partitioning *self, uint64_t glob_col_id);
uint64_t globalrow2localrow (Partitioning *self, uint64_t glob_row_id);
uint64_t edge2globalprocess (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);

// BUG for the row slicing we need to solve the problem of the transpose
typedef uint64_t (* PartitionFunction)(Partitioning *part, uint64_t glob_row_id, uint64_t glob_col_id);

inline int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

template<typename IT, typename VT>
Entry<IT, VT>* sortEntriesByOwner(const Entry<IT, VT>* entries, const int* owner, size_t nentries);

#endif // __DMMIO_DISTRIBUTED_H__
