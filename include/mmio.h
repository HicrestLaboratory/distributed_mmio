/* 
*   Matrix Market I/O library for ANSI C
*   See http://math.nist.gov/MatrixMarket for details.
*/

#ifndef MM_IO_H
#define MM_IO_H

#include <stdint.h>
#include <stdio.h>
#include <string>
#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[6];

int required_bytes_index(uint64_t maxval);

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx);
int mm_read_mtx_crd_size(FILE *f, uint64_t *M, uint64_t *N, uint64_t *nz);

int required_bytes_index(uint64_t maxval);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	    ((typecode)[0]=='M')

#define mm_is_sparse(typecode)	    ((typecode)[1]=='C')
#define mm_is_coordinate(typecode)  ((typecode)[1]=='C')
#define mm_is_dense(typecode)	      ((typecode)[1]=='A')
#define mm_is_array(typecode)	      ((typecode)[1]=='A')

#define mm_is_complex(typecode)	    ((typecode)[2]=='C')
#define mm_is_real(typecode)		    ((typecode)[2]=='R')
#define mm_is_pattern(typecode)	    ((typecode)[2]=='P')
#define mm_is_integer(typecode)     ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)   ((typecode)[3]=='S')
#define mm_is_general(typecode)	    ((typecode)[3]=='G')
#define mm_is_skew(typecode)	      ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)   ((typecode)[3]=='H')

#define mm_get_idx_bytes(typecode)  ((uint8_t)((unsigned char)((typecode)[4])))
#define mm_get_val_bytes(typecode)  ((uint8_t)((unsigned char)((typecode)[5])))

int mm_is_valid(MM_typecode matcode);		/* too complex for a macro */


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	    ((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	    ((*typecode)[1]='A')
#define mm_set_dense(typecode)	    mm_set_array(typecode)
#define mm_set_sparse(typecode)	    mm_set_coordinate(typecode)

#define mm_set_complex(typecode)    ((*typecode)[2]='C')
#define mm_set_real(typecode)       ((*typecode)[2]='R')
#define mm_set_pattern(typecode)    ((*typecode)[2]='P')
#define mm_set_integer(typecode)    ((*typecode)[2]='I')

#define mm_set_symmetric(typecode)  ((*typecode)[3]='S')
#define mm_set_general(typecode)    ((*typecode)[3]='G')
#define mm_set_skew(typecode)	      ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)  ((*typecode)[3]='H')

#define mm_set_idx_bytes(typecode, bytes)  ((*typecode)[4]=(char)((uint8_t)(bytes)))
#define mm_set_val_bytes(typecode, bytes)  ((*typecode)[5]=(char)((uint8_t)(bytes)))

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ',(*typecode)[3]='G',(*typecode)[4]=(*typecode)[5]='0')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/

#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF        12
#define MM_NOT_MTX				      13
#define MM_NO_HEADER			      14
#define MM_UNSUPPORTED_TYPE		  15
#define MM_LINE_TOO_LONG		    16
#define MM_COULD_NOT_WRITE_FILE	17


/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

				    ojbect 		sparse/   	data        storage 
						  		dense     	type        scheme

   string position:	 [0]        [1]			[2]         [3]

   Matrix typecode:  M(atrix)  C(oord)		R(eal)   	G(eneral)
						        A(array)	C(omplex)   H(ermitian)
											P(attern)   S(ymmetric)
								    		I(nteger)	K(kew)

 ***********************************************************************/

#define MM_MTX_STR		        "matrix"
#define MM_ARRAY_STR	        "array"
#define MM_DENSE_STR	        "array"
#define MM_COORDINATE_STR     "coordinate" 
#define MM_SPARSE_STR	        "coordinate"
#define MM_COMPLEX_STR	      "complex"
#define MM_REAL_STR		        "real"
#define MM_INT_STR		        "integer"
#define MM_GENERAL_STR        "general"
#define MM_SYMM_STR		        "symmetric"
#define MM_HERM_STR		        "hermitian"
#define MM_SKEW_STR		        "skew-symmetric"
#define MM_PATTERN_STR        "pattern"

enum MM_VAL_TYPE {
  MM_VAL_TYPE_REAL,
  MM_VAL_TYPE_INTEGER,
  MM_VAL_TYPE_PATTERN
};

struct Matrix_Metadata {
  MM_VAL_TYPE val_type;
  bool is_symmetric;
  std::string mm_header;
  std::string mm_header_body;
  uint8_t val_bytes;
};

/*  high level routines */

template<typename IT, typename VT>
struct Entry {
  IT row;
  IT col;
  VT val;
}; // For parsing

template<typename IT, typename VT>
struct CSR_local {
  IT 	nrows;
  IT 	ncols;
  IT 	nnz;
  IT 	*row_ptr;
  IT 	*col_idx;
  VT 	*val;
};

template<typename IT, typename VT>
struct COO_local {
  IT nrows;
  IT ncols;
  IT nnz;
  IT *row;
  IT *col;
  VT *val;
};

FILE *open_file_r(const char *filename);
int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[], double val[], MM_typecode matcode);

template<typename IT, typename VT>
int mm_read_mtx_crd_data(FILE *f, int nz, Entry<IT, VT> entries[], MM_typecode matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t val_bytes);

// template<typename IT, typename VT>
// int compare_entries_csr(const void *a, const void *b);

// template<typename IT, typename VT>
// Entry<IT, VT>* mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode);
bool is_file_extension_bmtx(std::string filename);

template<typename IT, typename VT>
int write_binary_matrix_market(FILE *f, COO_local<IT, VT> *coo, Matrix_Metadata *meta);

// Local CSR

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

template<typename IT, typename VT>
void Distr_MMIO_CSR_local_destroy(CSR_local<IT, VT>** csr);

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
CSR_local<IT, VT>* Distr_MMIO_CSR_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

// Local COO

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

template<typename IT, typename VT>
void Distr_MMIO_COO_local_destroy(COO_local<IT, VT>** csr);

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
COO_local<IT, VT>* Distr_MMIO_COO_local_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
Entry<IT, VT>* Distr_MMIO_COO_local_read_mpi(FILE *f, bool is_bmtx, int comm_size, int myrank, uint32_t *nentries, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

template<typename IT, typename VT>
int Distr_MMIO_COO_local_write(COO_local<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta);

template<typename IT, typename VT>
int Distr_MMIO_COO_local_write_f(COO_local<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta);

// ---------------------------------------------------------------------------------------------------------------------------------
//                                          Probably to move in a dedicated file
// ---------------------------------------------------------------------------------------------------------------------------------

#include<mpi.h>

struct ProcessGrid
{
    int grk; // global rank
    int gsz; // size of COMM_WORLD

    int rsz, csz, nsz; // row, column and node communicator sizes
    int rrk, crk, nrk; // row, column and node communicator ranks

    MPI_Comm world_comm, row_comm, col_comm, node_comm;

} typedef ProcessGrid;

ProcessGrid * make_process_grid(int row_size, int col_size, int node_size);
void print_process_grid(const ProcessGrid *grid, FILE* fp = stdout);

// ===================================================== Partitioning functions ===================================================
/* Partitioning functions to be used when reading in matrices
 *
 */

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


inline int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

#endif // MM_IO_H
