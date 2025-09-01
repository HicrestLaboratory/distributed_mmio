#include <algorithm>
#include <vector>
#include <unistd.h>
#include <mpi.h>
#include <ccutils/colors.h>
#include <ccutils/macros.h>

#include "../../include/distributed/distributed_mmio.h"

// TODO deleteme
#include <ccutils/macros.h>

#define DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template Entry<IT, VT>* sort_entries_by_owner(const Entry<IT, VT>* entries, const int* owner, size_t nentries); \
  template Entry<IT, VT>* mm_parse_file_distributed(FILE *f, int rank, int mpi_comm_size, IT &nrows, IT &ncols, IT &local_nnz, MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta);


DMMIO_ProcessGrid * make_process_grid(int row_size, int col_size, int node_size) {
    ASSERT(row_size > 0, "row_size must be > 0");
    ASSERT(col_size > 0, "col_size must be > 0");
    ASSERT(node_size > 0, "node_size must be > 0");
    ASSERT(row_size == col_size, "Currently only square grids are supported");

    DMMIO_ProcessGrid * grid = (DMMIO_ProcessGrid*)malloc(sizeof(DMMIO_ProcessGrid));

    grid->rsz = row_size;
    grid->csz = col_size;
    grid->nsz = node_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &(grid->grk));
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->gsz));
    grid->world_comm = MPI_COMM_WORLD;

    ASSERT((grid->gsz % (row_size*col_size) == 0), "Total number of processes must be a multiple of row_size * col_size");

    // Compute 3D coordinates from linear rank
    int i = grid->grk / (col_size * node_size);      // row index
    int remainder = grid->grk % (col_size * node_size);
    int j = remainder / node_size;                   // col index
    int k = remainder % node_size;                   // node index

    // --- Node communicator (same i,j, different k)
    int node_color = i * col_size + j; // unique per (i,j)
    int node_key   = k;
    MPI_Comm_split(MPI_COMM_WORLD, node_color, node_key, &(grid->node_comm));

    // --- Row communicator: (i, *, k) → fix i and k, vary j
    int row_color = i * node_size + k; // unique per (i,k)
    int row_key   = j;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &(grid->row_comm));

    // --- Column communicator: (*, j, k) → fix j and k, vary i
    int col_color = j * node_size + k; // unique per (j,k)
    int col_key   = i;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &(grid->col_comm));

    MPI_Barrier(grid->world_comm);

    MPI_Comm_rank(grid->row_comm, &(grid->rrk));
    MPI_Comm_size(grid->row_comm, &(grid->rsz));

    MPI_Comm_rank(grid->col_comm, &(grid->crk));
    MPI_Comm_size(grid->col_comm, &(grid->csz));

    MPI_Comm_rank(grid->node_comm, &(grid->nrk));
    MPI_Comm_size(grid->node_comm, &(grid->nsz));

//     if (grid->rsz != row_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
//     if (grid->csz != col_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
//     if (grid->nsz != node_size) MPI_Abort(MPI_COMM_WORLD, __LINE__);

    return grid;
}

void print_process_grid(const DMMIO_ProcessGrid *grid, FILE* fp) {
	if (grid->grk==0) {
        fprintf(fp, "========================\n");
        fprintf(fp, " DMMIO_ProcessGrid Details \n");
        fprintf(fp, "========================\n");
        fprintf(fp, "Total processes:\t %d\n", grid->gsz);
        fprintf(fp, "row size:\t %d\n", grid->rsz);
        fprintf(fp, "col size:\t %d\n", grid->csz);
        fprintf(fp, "node size:\t %d\n", grid->nsz);

    }
    MPI_Barrier(MPI_COMM_WORLD);
	sleep(1);

    for (int i=0; i<grid->gsz; i++) {
        if (grid->grk == i) {
            fprintf(fp, "----- Process %d -----\n", grid->grk);
            fprintf(fp, "Rank:\t %d\n", grid->grk);
            fprintf(fp, "row rank:\t %d\n", grid->rrk);
            fprintf(fp, "col rank:\t %d\n", grid->crk);
            fprintf(fp, "node rank:\t %d\n", grid->nrk);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    sleep(1);
    if (grid->grk == 0) fprintf(fp, "========================\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

// ==================================================== Partitioning functions ==================================================
/*  Functions and macros to compute the partitioning
 *
 *  The partitioning could be hyerarcycal ...to explain...
 *
 */

void init_mypart(DMMIO_Partitioning *self) {
    self->type_str = NULL;

    self->grid = NULL;

    self->op = DMMIO_OP_NONE;

    self->global_matrix_rows = 0;
    self->global_matrix_cols = 0;
    self->group_matrix_rows = 0;
    self->group_matrix_cols = 0;
    self->local_matrix_rows = 0;
    self->local_matrix_cols = 0;

    self->edge2group = NULL;
    self->edge2nodeprocess = NULL;
    self->globalcol2groupcol = NULL;
    self->globalrow2grouprow = NULL;
    self->groupcol2localcol = NULL;
    self->grouprow2localrow = NULL;
}

void set_mypart_str (DMMIO_Partitioning *self) {
	switch (self->type) {
		case DMMIO_PARTITIONING_RSLICING:
			self->type_str = "partRslicing";
			break;
		case DMMIO_PARTITIONING_BCYCLE:
			self->type_str = "partBcycle";
			break;
        case DMMIO_PARTITIONING_NAIVE_CYCLE:
            self->type_str = "partNaiveCycle";
            break;
        case DMMIO_PARTITIONING_BCYCLE_CYCLE:
            self->type_str = "partBcycleCycle";
            break;
        case DMMIO_PARTITIONING_RSLICING_CYCLE:
			self->type_str = "partRslicingCycle";
			break;
		default:
			self->type_str = "partNaive";
			break;
    }
}

void set_mypart_transpose(DMMIO_Partitioning *self, char op) {
  // ASSERT((op=='l')||(op=='r'), "Error in function %s, op must be 'l' (left) or 'r' (right).", __func__);
  self->op = (op=='l') ? DMMIO_OP_NONE : DMMIO_OP_TRANSPOSE;
}

void set_mypart_type (DMMIO_Partitioning *self, DMMIO_PARTITIONING type, char op) {
  init_mypart(self);
	self->type = type;
	set_mypart_str(self);
  set_mypart_transpose(self, op);
}

void set_mypart_grid (DMMIO_Partitioning *self, DMMIO_ProcessGrid* grid) {
	self->grid = grid;
}

void set_mypart_globaldim (DMMIO_Partitioning *self, uint64_t n, uint64_t m) {
	self->global_matrix_rows = n;
	self->global_matrix_cols = m;
}

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

uint64_t edge2proc_1d_naive(uint64_t idx,
                       uint64_t dimtosplit,
                       int nprocs)
{
	int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs);
    return (idx / chunk_size );
}

uint64_t edge2proc_1d_cycle(uint64_t idx,
                       uint64_t dimtosplit,
                       int nprocs)
{
	int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs*nprocs);
    return ( (idx/chunk_size) % nprocs );
}

// Defined in the header
// typedef uint64_t (*Edge2proc2dFunction)(uint64_t i, uint64_t j,
// 										uint64_t nrows, uint64_t ncols,
// 										int procs_per_row, int procs_per_col);

uint64_t edge2proc_2d_naive(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
	int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
	int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);
    return ( (i/rows_chunk_size)*procs_per_row + j/cols_chunk_size );
}

uint64_t edge2proc_2d_rowslicing(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);

    int process_col_coord = j/cols_chunk_size;
    int process_row_coord = (i/rows_chunk_size) % procs_per_col;

	return ( process_row_coord*procs_per_row + process_col_coord);
}

uint64_t edge2proc_2d_rowslicing_transpose(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
    int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
    int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

    int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    int process_row_coord = i/rows_chunk_size;
	return ( process_row_coord*procs_per_row + process_col_coord);
}

uint64_t edge2proc_2d_blockcycle(uint64_t i, uint64_t j,
                       uint64_t nrows, uint64_t ncols,
                       int procs_per_row, int procs_per_col)
{
	int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
	int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

	int process_row_coord = (i/rows_chunk_size) % procs_per_col;
	int process_col_coord = (j/cols_chunk_size) % procs_per_row;
    return ( process_row_coord*procs_per_row + process_col_coord);
}


uint64_t edge2proc_2d_1d(uint64_t i, uint64_t j,
                         uint64_t m, uint64_t n,
                         DMMIO_ProcessGrid * grid, int transpose)
{
    ASSERT((n % grid->csz == 0), "pc %d does not divide n %lu\n", grid->csz, n); // Process grid must evenly divide matrix dims
    ASSERT((m % (grid->rsz * grid->nsz) == 0), "pr*pz %d does not divide m %lu\n", grid->nsz * grid->nsz, m);

	int rpg = m / grid->rsz;
    int cpg = n / grid->csz;


    int gid = (i / rpg) * grid->csz + j / cpg;

    int rpp = rpg / grid->nsz;
    int intragroup_id = (i % rpg) / rpp;

    int pid = gid * grid->nsz + intragroup_id;

#if DEBUG_PARTITION
    printf("(%lu, %lu) mapped to %d. rpg:%d, cpg:%d, gid:%d, rpp:%d, ig_id: %d\n",
                    i, j, pid, rpg, cpg, gid, rpp, intragroup_id);
    FLUSH_WAIT(0.5);
#endif

    ASSERT((pid < grid->gsz), "pid is %d, must be < %d\n", pid, grid->gsz);

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

// Others

void set_mypart_groupdim (DMMIO_Partitioning *self) {
	ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

	self->group_matrix_rows = PART_CHUNK_SIZE(self->global_matrix_rows, (self->grid)->rsz);
	self->group_matrix_cols = PART_CHUNK_SIZE(self->global_matrix_cols, (self->grid)->csz);
}

void set_mypart_localdim (DMMIO_Partitioning *self) {
	ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

    ASSERT(((self->group_matrix_rows!=0)&&(self->group_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tgroup_matrix_rows: %lu\n\tgroup_matrix_cols: %lu\n", \
		__func__, self->group_matrix_rows, self->group_matrix_cols);

    // BUG, tmp fix
//     if (self->op == 'l') {
//         self->local_matrix_rows = PART_CHUNK_SIZE(self->group_matrix_rows, (self->grid)->pz);
//         self->local_matrix_cols = self->group_matrix_cols;
//     } else {
//         self->local_matrix_rows = self->group_matrix_rows;
//         self->local_matrix_cols = PART_CHUNK_SIZE(self->group_matrix_cols, (self->grid)->pz);
//     }
    self->local_matrix_rows = PART_CHUNK_SIZE(self->group_matrix_rows, (self->grid)->nsz);
    self->local_matrix_cols = self->group_matrix_cols;
}

void set_mypart_functions (DMMIO_Partitioning *self) {
	ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

	ASSERT(((self->global_matrix_rows!=0)&&(self->global_matrix_cols!=0)), \
		"%s call before matrix dim are set:\n\tglobal_matrix_rows: %lu\n\tglobal_matrix_cols: %lu\n", \
		__func__, self->global_matrix_rows, self->global_matrix_cols);

	ASSERT(((self->group_matrix_rows!=0)&&(self->group_matrix_cols!=0)), \
		"%s call before group matrix dim are set:\n\tgroup_matrix_rows: %lu\n\tgroup_matrix_cols: %lu\n", \
		__func__, self->group_matrix_rows, self->group_matrix_cols);


	switch (self->type) {
		case DMMIO_PARTITIONING_BCYCLE: {
			// LOAD RSLICING PARTITIONING
			self->edge2group = edge2proc_2d_blockcycle;
			self->edge2nodeprocess = edge2proc_1d_naive;
			self->globalcol2groupcol = globalindex2localindex_cycle;
			self->globalrow2grouprow = globalindex2localindex_cycle;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case DMMIO_PARTITIONING_BCYCLE_CYCLE: {
			// LOAD RSLICING PARTITIONING
			self->edge2group = edge2proc_2d_blockcycle;
			self->edge2nodeprocess = edge2proc_1d_cycle;
			self->globalcol2groupcol = globalindex2localindex_cycle;
			self->globalrow2grouprow = globalindex2localindex_cycle;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_cycle;
			break;
		}

		case DMMIO_PARTITIONING_RSLICING: {
			// DMMIO_PARTITIONING_RSLICING and DMMIO_PARTITIONING_NAIVE use the same column partitioning
			self->edge2nodeprocess = edge2proc_1d_naive;

            if (self->op == 'l') {
                self->edge2group = edge2proc_2d_rowslicing;
                self->globalcol2groupcol = globalindex2localindex_naive;
                self->globalrow2grouprow = globalindex2localindex_cycle;
            } else {
                self->edge2group = edge2proc_2d_rowslicing_transpose;
                self->globalcol2groupcol = globalindex2localindex_cycle;
                self->globalrow2grouprow = globalindex2localindex_naive;
            }

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case DMMIO_PARTITIONING_RSLICING_CYCLE: {
			// DMMIO_PARTITIONING_RSLICING and DMMIO_PARTITIONING_NAIVE use the same column partitioning
			self->edge2nodeprocess = edge2proc_1d_cycle;

            if (self->op == 'l') {
                self->edge2group = edge2proc_2d_rowslicing;
                self->globalcol2groupcol = globalindex2localindex_naive;
                self->globalrow2grouprow = globalindex2localindex_cycle;
            } else {
                self->edge2group = edge2proc_2d_rowslicing_transpose;
                self->globalcol2groupcol = globalindex2localindex_cycle;
                self->globalrow2grouprow = globalindex2localindex_naive;
            }

			self->groupcol2localcol = (self->op=='l') ? globalindex2localindex_naive : globalindex2localindex_cycle ;
			self->grouprow2localrow = (self->op=='l') ? globalindex2localindex_cycle : globalindex2localindex_naive ;
			break;
		}

		case DMMIO_PARTITIONING_NAIVE: {
			// DMMIO_PARTITIONING_RSLICING and DMMIO_PARTITIONING_NAIVE use the same column partitioning
			self->edge2group = edge2proc_2d_naive;
			self->edge2nodeprocess = edge2proc_1d_naive;
			self->globalcol2groupcol = globalindex2localindex_naive;
			self->globalrow2grouprow = globalindex2localindex_naive;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_naive;
			break;
		}

		case DMMIO_PARTITIONING_NAIVE_CYCLE: {
			// DMMIO_PARTITIONING_RSLICING and DMMIO_PARTITIONING_NAIVE use the same column partitioning
			self->edge2group = edge2proc_2d_naive;
			self->edge2nodeprocess = edge2proc_1d_cycle;
			self->globalcol2groupcol = globalindex2localindex_naive;
			self->globalrow2grouprow = globalindex2localindex_naive;

			self->groupcol2localcol = globalindex2localindex_naive;
			self->grouprow2localrow = globalindex2localindex_cycle;
			break;
		}
	}
}

// Functions from edge to process
uint64_t edge2group (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->edge2group(glob_row_id, glob_col_id, self->global_matrix_rows, self->global_matrix_cols, (self->grid)->csz, (self->grid)->rsz) );
}

uint64_t edge2nodeprocess (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     uint64_t idxtosplit = (self->op=='l') ? (globalrow2grouprow(self, glob_row_id)) : (globalcol2groupcol(self, glob_col_id)) ;
//     uint64_t dimtosplit = (self->op=='l') ? self->group_matrix_rows : self->group_matrix_cols ;
    // BUG, tmp fix
    uint64_t idxtosplit = (globalrow2grouprow(self, glob_row_id));
    uint64_t dimtosplit = self->group_matrix_rows;
	return( self->edge2nodeprocess(idxtosplit, dimtosplit, (self->grid)->nsz) );
}

// Function from upper index to inner index
uint64_t globalcol2groupcol (DMMIO_Partitioning *self, uint64_t glob_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->globalcol2groupcol(glob_col_id, self->global_matrix_cols, (self->grid)->csz) );
}

uint64_t globalrow2grouprow (DMMIO_Partitioning *self, uint64_t glob_row_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
	return( self->globalrow2grouprow(glob_row_id, self->global_matrix_rows, (self->grid)->rsz) );
}

uint64_t groupcol2localcol (DMMIO_Partitioning *self, uint64_t grp_col_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     int ncolsplit = (self->op=='l') ? 1 : ((self->grid)->pz) ;
    int ncolsplit = 1; // BUG, tmp fix
	return( self->groupcol2localcol(grp_col_id, self->group_matrix_cols, ncolsplit) );
}
uint64_t grouprow2localrow (DMMIO_Partitioning *self, uint64_t grp_row_id) {
#ifndef SKIP_SETPARTFUNC_ASSERT
	ASSERT((self->grouprow2localrow!=NULL), "%s call before set_mypart_functions\n", __func__);
#endif
//     int nrowsplit = (self->op=='l') ? ((self->grid)->pz) : 1 ;
    int nrowsplit = ((self->grid)->nsz); // BUG, tmp fix
	return( self->grouprow2localrow(grp_row_id, self->group_matrix_rows, nrowsplit) );
}

// Composed function: from global index to local index and from edge to global rank
uint64_t globalcol2localcol (DMMIO_Partitioning *self, uint64_t glob_col_id) {
	uint64_t grp_col_id = globalcol2groupcol (self, glob_col_id);
	return(  groupcol2localcol(self, grp_col_id) );
}
uint64_t globalrow2localrow (DMMIO_Partitioning *self, uint64_t glob_row_id) {
	uint64_t grp_row_id = globalrow2grouprow (self, glob_row_id);
	return( grouprow2localrow(self, grp_row_id) );
}

uint64_t edge2globalprocess (DMMIO_Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) {
	int gid = edge2group(self, glob_row_id, glob_col_id); // BUG for the row slicing we need to solve the problem of the transpose
	int intragroup_id = edge2nodeprocess(self, glob_row_id, glob_col_id);
	int pid = gid * ((self->grid)->nsz) + intragroup_id;

#if DEBUG_PARTITION
    printf("%s: (%lu, %lu) mapped to %d. gid: %d, ig_id: %d\n",
                    self->my_part_str, glob_row_id, glob_col_id, pid, gid, intragroup_id);
    FLUSH_WAIT(0.5);
#endif
    return(pid);
}


// Sort entries by owner, producing a new sorted vector
template<typename IT, typename VT>
Entry<IT, VT>* sort_entries_by_owner(const Entry<IT, VT>* entries, const int* owner, size_t nentries) {
    // Combine entries and owner into a vector of pairs
    std::vector<std::pair<int, Entry<IT, VT>>> combined(nentries);
    for (size_t i = 0; i < nentries; ++i) {
        combined[i] = { owner[i], entries[i] };
    }

    // Sort by owner
    std::sort(combined.begin(), combined.end(),
              [](const std::pair<int, Entry<IT, VT>>& a, const std::pair<int, Entry<IT, VT>>& b) {
                  return a.first < b.first;
              });

    // Allocate new array for sorted entries
    Entry<IT, VT>* sorted_entries = (Entry<IT, VT>*)malloc(nentries * sizeof(Entry<IT, VT>));
    for (size_t i = 0; i < nentries; ++i) {
        sorted_entries[i] = combined[i].second;
    }

    return sorted_entries;
}

template<typename IT, typename VT>
Entry<IT, VT>* mm_parse_file_distributed(
  FILE *f,
  int rank, int mpi_comm_size,
  IT &nrows, IT &ncols, IT &local_nnz,
  MM_typecode *matcode, bool is_bmtx, DMMIO_Matrix_Metadata* meta
) {
  ASSERT(is_bmtx, "Distributed read of non-binary MTX files is not supported yet") // TODO implement

  IT nentries = 0, nnz_upperbound = 0;
  if (f == NULL) {
    printf("File pointer is NULL\n");
    return NULL;
  }
  int err = mm_parse_header<IT, VT>(f, nrows, ncols, nentries, nnz_upperbound, matcode, is_bmtx, meta);
  if (err != 0) {
    printf("Could not parse matrix header (error code: %d).\n", err);
    fclose(f);
    return NULL;
  }
  
  long int pos = ftell(f);
  uint16_t line_size = meta->is_pattern ? (2 * meta->index_bytes) : (2 * meta->index_bytes + meta->value_bytes);
  nentries = (rank < nnz_upperbound % mpi_comm_size) ? (nnz_upperbound / mpi_comm_size + 1) : (nnz_upperbound / mpi_comm_size);
  uint32_t to_skip = (rank < nnz_upperbound % mpi_comm_size) ? (nentries * rank) : (nentries * rank + nnz_upperbound % mpi_comm_size);
  Entry<IT, VT> *entries = (Entry<IT, VT> *)malloc(nentries * sizeof(Entry<IT, VT>));

  if (fseek(f, to_skip*line_size, SEEK_CUR) != 0) {
    perror("fseek failed");
    fclose(f);
    return(NULL);
  }

  long int new_pos = ftell(f);
  fprintf(stdout, "[DEBUG at line %d] process %d: %u nentries, %u to skip, %ld is the starting position and %ld is the current position (%u B)\n", __LINE__, rank, nentries, to_skip, pos, new_pos, to_skip*line_size);

  err = mm_read_mtx_crd_data<IT, VT>(f, nentries, entries, matcode, is_bmtx, meta->index_bytes, meta->value_bytes);
  fclose(f);
  if (err != 0) {
    printf("Could not parse matrix data (error code: %d).\n", err);
    free(entries);
    return NULL;
  }

  local_nnz = mm_duplicate_entries_for_symmetric_matrices(entries, nentries, meta);

  return entries;
}

DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)