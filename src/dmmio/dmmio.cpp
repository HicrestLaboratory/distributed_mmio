#include <unistd.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <ccutils/colors.h>
#include <ccutils/macros.h>
#include <ccutils/mpi/mpi_macros.h>

#include "../../include/mmio/mmio.h"
#include "../../include/mmio/io.h"
#include "../../include/dmmio/dmmio.h"
#include "../../include/dmmio/dio.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
using Operation = dmmio::Operation;
using PartitioningType = dmmio::PartitioningType;
using Partitioning = dmmio::Partitioning;
using ProcessGrid = dmmio::ProcessGrid;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using DCOO = dmmio::DCOO<IT, VT>;

#define DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template DCOO<IT, VT>* dmmio::DCOO_read(const char *filename, int mpi_comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size, PartitioningType partitioning_type, Operation op, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template DCOO<IT, VT>* dmmio::DCOO_read_f(FILE* f, int comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size, PartitioningType part_type, Operation op, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template void dmmio::DCOO_destroy(DCOO<IT, VT>** dcoo);


namespace dmmio {

  template<typename IT, typename VT>
  Entry<IT, VT>* sortEntriesByOwner(const Entry<IT, VT>* entries, const int* owner, size_t nentries) {
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
    
  Partitioning* Partitioning_create(int matrix_rows, int matrix_cols, int grid_rows, int grid_cols, int grid_node_size, PartitioningType partitioning_type, Operation operation) {
    Partitioning *partitioning = (Partitioning*)malloc(sizeof(Partitioning));
    ProcessGrid *grid = dmmio::io::ProcessGrid_create(grid_rows, grid_cols, grid_node_size);

    dmmio::io::set_partitioning_type(partitioning, partitioning_type, operation);
    dmmio::io::set_partitioning_grid(partitioning, grid);
    dmmio::io::set_partitioning_global_dim(partitioning, matrix_rows, matrix_cols);
    dmmio::io::set_partitioning_group_dim(partitioning);
    dmmio::io::set_partitioning_local_dim(partitioning);
    dmmio::io::set_partitioning_functions(partitioning);

    return partitioning;
  }

  void Partitioning_destroy(Partitioning **partitioning) {
    if (partitioning != NULL && *partitioning != NULL) {
      if ((*partitioning)->grid != NULL) {
        MPI_Comm_free(&((*partitioning)->grid->row_comm));
        MPI_Comm_free(&((*partitioning)->grid->col_comm));
        MPI_Comm_free(&((*partitioning)->grid->node_comm));
        free((*partitioning)->grid);
      }
    }
    if (partitioning != NULL) free(partitioning);
  }

  template<typename IT, typename VT>
  DCOO<IT, VT>* DCOO_read(
    const char *filename,
    int mpi_comm_size, int rank,
    int grid_rows, int grid_cols, int grid_node_size,
    PartitioningType partitioning_type, Operation op,
    bool expl_val_for_bin_mtx, Matrix_Metadata* meta
  ) {
    return DCOO_read_f<IT, VT>(
      mmio::io::open_file_r(filename),
      mpi_comm_size, rank,
      grid_rows, grid_cols, grid_node_size,
      partitioning_type, op,
      mmio::io::mm_is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta
    );
  }

  template<typename IT, typename VT>
  DCOO<IT, VT>* DCOO_read_f(
    FILE *f,
    int mpi_comm_size, int rank,
    int grid_rows, int grid_cols, int grid_node_size,
    PartitioningType partitioning_type, Operation op,
    bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta
  ) {
    IT nrows, ncols, nnz;
    MM_typecode matcode;
    Entry<IT, VT> *entries = dmmio::io::mm_parse_file_distributed<IT, VT>(f, rank, mpi_comm_size, nrows, ncols, nnz, &matcode, is_bmtx, meta);
    if (entries == NULL) return NULL;
    Partitioning *partitioning = Partitioning_create(nrows, ncols, grid_rows, grid_cols, grid_node_size, partitioning_type, op);
    DCOO<IT, VT> *dcoo = (DCOO<IT, VT>*)malloc(sizeof(DCOO<IT, VT>));
    dcoo->partitioning = partitioning;

    int *owner = (int*)malloc(sizeof(int)*nnz);
    for (int i=0; i<nnz; i++) owner[i] = dmmio::io::edge2globalprocess(partitioning, entries[i].row, entries[i].col);

    #ifdef DEBUG
    MPI_ALL_PRINT(
      for (int i=0; i<nnz; i++) {
        fprintf(fp, "\t%lu, %lu, %lu --> %d\n", entries[i].row, entries[i].col, entries[i].val, owner[i]);
      }
      fprintf(fp, "\n");
    )
    #endif

    // Sort the entries according to the owner process and rebuild the new owner vector
    Entry<IT, VT>* sorted_entries = sortEntriesByOwner<IT, VT>(entries, owner, nnz);
    free(entries);
    free(owner);

    owner = (int*)malloc(sizeof(int)*nnz);
    for (int i=0; i<nnz; i++) owner[i] = dmmio::io::edge2globalprocess(partitioning, sorted_entries[i].row, sorted_entries[i].col);

    #ifdef DEBUG
    MPI_ALL_PRINT(
      fprintf(fp, "Sorted entries:\n");
      for (int i=0; i<nnz; i++) {
          fprintf(fp, "\t%lu, %lu, %lu --> %d\n", sorted_entries[i].row, sorted_entries[i].col, sorted_entries[i].val, owner[i]);
      }
      fprintf(fp, "\n");
    )
    #endif

    int* counts_send        = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* counts_recv        = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* displacements_send = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* displacements_recv = (int*)malloc(sizeof(int)*mpi_comm_size);

    for (int i=0; i<mpi_comm_size; i++) counts_send[i] = 0;
    for (int i=0; i<nnz; i++) counts_send[owner[i]] += sizeof(Entry<IT, VT>);

    displacements_send[0] = 0;
    for (int i=1; i<mpi_comm_size; i++) displacements_send[i] = displacements_send[i-1] + counts_send[i-1];

    MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

    displacements_recv[0] = 0;
    for (int i = 1; i < mpi_comm_size; i++) displacements_recv[i] = displacements_recv[i-1] + counts_recv[i-1];
    int total_recv = (displacements_recv[mpi_comm_size-1] + counts_recv[mpi_comm_size-1]) / sizeof(Entry<IT, VT>);
    Entry<IT, VT>* recv_entries = (Entry<IT, VT>*)malloc(total_recv * sizeof(Entry<IT, VT>));

  #ifdef DEBUG_ALLTOALLV
    MPI_ALL_PRINT(
        fprintf(fp, "Rank %d of %d\n", world_rank, mpi_comm_size);
        fprintf(fp, "nnz: %d | total_recv: %d\n\n", nnz, total_recv);

        // Print counts_send
        fprintf(fp, "\tcounts_send:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", counts_send[i]);
        fprintf(fp, "\n");

        // Print displacements_send
        fprintf(fp, "\tdisplacements_send:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", displacements_send[i]);
        fprintf(fp, "\n");

        // Print counts_recv
        fprintf(fp, "\tcounts_recv:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", counts_recv[i]);
        fprintf(fp, "\n");

        // Print displacements_recv
        fprintf(fp, "\tdisplacements_recv:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", displacements_recv[i]);
        fprintf(fp, "\n\n");

        // Optionally, print the sorted entries being sent
        fprintf(fp, "\tEntries being sent (sorted by owner):\n");
        for (int i = 0; i < nnz; i++) {
            fprintf(fp, "\t\t[%3d] -> row: %lu, col: %lu, val: %lu\n",
                    owner[i], sorted_entries[i].row, sorted_entries[i].col, sorted_entries[i].val);
        }
        fprintf(fp, "\n")
    )
  #endif

    MPI_Alltoallv(sorted_entries,
                  counts_send,
                  displacements_send,
                  MPI_BYTE,
                  recv_entries,
                  counts_recv,
                  displacements_recv,
                  MPI_BYTE,
                  MPI_COMM_WORLD);

  #define DEBUG_RESULT
  #ifdef DEBUG_RESULT

    // Define the dbg_entry (which embed an entry and the entry owner)
    typedef struct dbg_entry {
        Entry<IT, VT> entry;
        int owner;
    } DbgEntry;
    DbgEntry* dbg_entries = (DbgEntry*)malloc(total_recv * sizeof(DbgEntry));

    for (int i=0; i<total_recv; i++) {
        dbg_entries[i].owner = dmmio::io::edge2globalprocess(partitioning, recv_entries[i].row, recv_entries[i].col);
        memcpy(&(dbg_entries[i].entry), &(recv_entries[i]), sizeof(Entry<IT, VT>));
    }

    #ifdef DEBUG
    MPI_ALL_PRINT(
      fprintf(fp, "Receved entries:\n");
      for (int i=0; i<total_recv; i++) {
          fprintf(fp, "\t%lu, %lu, %lu --> %d\n", dbg_entries[i].entry.row, dbg_entries[i].entry.col, dbg_entries[i].entry.val, dbg_entries[i].owner);
      }
      fprintf(fp, "\n")
    )
    #endif

    // Allghaterv to collect all the coordinates on a single process and perform a debug print
    int* gather_counts = (int*)malloc(sizeof(int) * mpi_comm_size);
    int* gather_displs = (int*)malloc(sizeof(int) * mpi_comm_size);

    MPI_Allgather(&total_recv, 1, MPI_INT, gather_counts, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < mpi_comm_size; i++) gather_counts[i] *= sizeof(DbgEntry);

    gather_displs[0] = 0;
    for (int i = 1; i < mpi_comm_size; i++) gather_displs[i] = gather_displs[i - 1] + gather_counts[i - 1];
    int total_entries = (gather_displs[mpi_comm_size - 1] + gather_counts[mpi_comm_size - 1]) / sizeof(DbgEntry) ;

    DbgEntry* all_entries = (DbgEntry*)malloc(total_entries * sizeof(DbgEntry));

    MPI_Allgatherv(
        dbg_entries,                   // send buffer
        total_recv * sizeof(DbgEntry), // send count
        MPI_BYTE,                      // type
        all_entries,                   // receive buffer
        gather_counts,                 // receive counts
        gather_displs,                 // displacements
        MPI_BYTE,                      // type
        MPI_COMM_WORLD
    );

  #ifdef DEBUG_ALLGHATERV
    MPI_PROCESS_PRINT( MPI_COMM_WORLD, 0,
        fprintf(stdout, "\n==================== ALLGATHERV CHECK ====================\n");
        fprintf(stdout, "Rank %d: gathered %d entries (total %d):\n", world_rank, total_entries, total_entries);
        for (int i = 0; i < total_entries; i++) {
            fprintf(stdout, "\trow: %lu, col: %lu, val: %lu, owner: %d\n",
                    all_entries[i].entry.row, all_entries[i].entry.col, all_entries[i].entry.val, all_entries[i].owner);
        }
        fprintf(stdout, "\n")
    )
  #endif

    // Prints coords on a file to be printed by using 'plotPartitioningCheck.py' script
    if (rank == 0) {
        char csvname[200];
        sprintf(csvname, "checktest_%d_%c.csv", partitioning_type, op);
        FILE *statfile = fopen(csvname, "w");
        fprintf(statfile, "rank,rowid,colid,val\n");

        for (int k=0; k<total_entries; k++) {
            fprintf(statfile, "%4d,%4lu,%4lu,%lu\n",
                        all_entries[k].owner, all_entries[k].entry.row, all_entries[k].entry.col, all_entries[k].entry.val);
        }
        fclose(statfile);
    }
  #endif

    return dcoo;
  }

  template<typename IT, typename VT>
  void DCOO_destroy(DCOO<IT, VT>** dcoo) {
    if (dcoo != NULL && *dcoo != NULL) {
      mmio::COO_destroy(&((*dcoo)->coo));
      Partitioning_destroy(&((*dcoo)->partitioning));
    }
    // FIXME: double free or corruption (fasttop)
    // if (dcoo != NULL) { 
    //   free(*dcoo);
    //   *dcoo = NULL;
    // }
  }

} // namespace dmmio

DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)