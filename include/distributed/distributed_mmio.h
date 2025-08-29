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


/********************* Utility Functions ***************************/

inline int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

/********************* Partitioning Functions (API) ***************************/

typedef uint64_t (*Edge2proc1dFunction)(uint64_t idx,
										uint64_t dimtosplit,
										int nprocs);

typedef uint64_t (*Edge2proc2dFunction)(uint64_t i,         uint64_t j,
										uint64_t nrows,     uint64_t ncols,
										int procs_per_row,  int procs_per_col);

typedef uint64_t (*GlobalIdx2LocalIdx)(uint64_t globalid, uint64_t globalsize, int nprocs);


/********************* Distributed Structs and Enums ***************************/

// TODO 
/// @brief Please refer to the README for a visual explanation
typedef enum {
    DMMIO_PARTITIONING_NAIVE,         // 0
    DMMIO_PARTITIONING_BCYCLE,        // 1
    DMMIO_PARTITIONING_RSLICING,      // 2
    DMMIO_PARTITIONING_NAIVE_CYCLE,   // 3
    DMMIO_PARTITIONING_BCYCLE_CYCLE,  // 4
    DMMIO_PARTITIONING_RSLICING_CYCLE // 5
} DMMIO_PARTITIONING;

typedef enum {
    DMMIO_OP_NONE,      // 0
    DMMIO_OP_TRANSPOSE, // 1
} DMMIO_OP;

struct DMMIO_ProcessGrid {
    int grk; // Global rank
    int gsz; // Size of COMM_WORLD

    int rrk, crk, nrk; // Row, Column and Node communicator ranks
    int rsz, csz, nsz; // Row, Column and Node communicator sizes

    MPI_Comm world_comm, row_comm, col_comm, node_comm;
};

struct DMMIO_Partitioning {
    DMMIO_PARTITIONING type;
    const char *type_str;
    DMMIO_OP op;

    DMMIO_ProcessGrid * grid;

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
};

template<typename IT, typename VT>
struct DCOO {
    DMMIO_Partitioning* partitioning;
    COO<IT, VT>* coo;
};


/********************* Partitioning Functions (API) cont. ***************************/

// BUG for the row slicing we need to solve the problem of the transpose
typedef uint64_t (*PartitionFunction)(DMMIO_Partitioning *part, uint64_t glob_row_id, uint64_t glob_col_id);


/********************* Partitioning Functions (Implementations) ***************************/

uint64_t edge2proc_2d_1d(uint64_t i, uint64_t j,
                         uint64_t m, uint64_t n,
                         DMMIO_ProcessGrid * grid, int transpose);


/********************* Struct Functions ***************************/

DMMIO_ProcessGrid * make_process_grid(int row_size, int col_size, int node_size);
void print_process_grid(const DMMIO_ProcessGrid *grid, FILE* fp = stdout);

void compute_local_matrix_dims(const DMMIO_Partitioning* part,
                               const uint64_t global_nrows,
                               const uint64_t global_ncols,
                               uint64_t * loc_nrows,
                               uint64_t * loc_ncols);

void set_mypart_type (DMMIO_Partitioning *self, DMMIO_PARTITIONING partype, char op = 'l');
void set_mypart_grid (DMMIO_Partitioning *self, DMMIO_ProcessGrid* grid);
void set_mypart_globaldim (DMMIO_Partitioning *self, uint64_t n, uint64_t m);
void set_mypart_groupdim (DMMIO_Partitioning *self);
void set_mypart_localdim (DMMIO_Partitioning *self);
void set_mypart_functions (DMMIO_Partitioning *self);

// Functions from edge to process
uint64_t edge2group (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);
uint64_t edge2nodeprocess (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);

// Function from upper index to inner index
uint64_t globalcol2groupcol (DMMIO_Partitioning *self, uint64_t glob_col_id);
uint64_t globalrow2grouprow (DMMIO_Partitioning *self, uint64_t glob_row_id);
uint64_t groupcol2localcol  (DMMIO_Partitioning *self, uint64_t grp_col_id);
uint64_t grouprow2localrow  (DMMIO_Partitioning *self, uint64_t grp_row_id);

// Composed function: from global index to local index and from edge to global rank
uint64_t globalcol2localcol (DMMIO_Partitioning *self, uint64_t glob_col_id);
uint64_t globalrow2localrow (DMMIO_Partitioning *self, uint64_t glob_row_id);
uint64_t edge2globalprocess (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);


/********************* DMMIO API ***************************/

template<typename IT, typename VT>
Entry<IT, VT>* sort_entries_by_owner(const Entry<IT, VT>* entries, const int* owner, size_t nentries);

template<typename IT, typename VT>
Entry<IT, VT>* mm_parse_file_distributed(FILE *f, int rank, int mpi_comm_size, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta);

template<typename IT, typename VT>
DCOO<IT, VT>* DMMIO_DCOO_read(
  const char *filename,
  int mpi_comm_size, int rank,
  int grid_rows, int grid_cols, int grid_node_size,
  DMMIO_PARTITIONING partitioning_type, DMMIO_OP op=DMMIO_OP_NONE,
  bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL
);

template<typename IT, typename VT>
DCOO<IT, VT>* DMMIO_DCOO_read_f(
  FILE *f,
  int comm_size, int rank,
  int grid_rows, int grid_cols, int grid_node_size,
  DMMIO_PARTITIONING partitioning_type, DMMIO_OP op=DMMIO_OP_NONE,
  bool is_bmtx=false, bool expl_val_for_bin_mtx=false, DMMIO_Matrix_Metadata* meta=NULL
);


#endif // __DMMIO_DISTRIBUTED_H__
