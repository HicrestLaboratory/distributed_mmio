#include <unistd.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>
#include <cassert>
#include <cstring>
#include <ccutils/colors.h>
#include <ccutils/macros.h>
#include <ccutils/mpi/mpi_macros.h>

#include "../../include/mmio/mmio.h"
#include "../../include/mmio/io.h"
#include "../../include/dmmio/dmmio.h"
#include "../../include/dmmio/dio.h"
#include "../../include/dmmio/partitioning.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
using Operation = dmmio::Operation;
using PartitioningType = dmmio::PartitioningType;
using Partitioning = dmmio::Partitioning;
using ProcessGrid = dmmio::ProcessGrid;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using DCOO = dmmio::DCOO<IT, VT>;
template<typename IT, typename VT> using COO = mmio::COO<IT, VT>;

#define DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template DCOO<IT, VT>* dmmio::DCOO_read(const char *filename, int mpi_comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size, PartitioningType partitioning_type, Operation op, bool expl_val_for_bin_mtx, Matrix_Metadata* meta, int padding, bool permute, IT * perm_vec); \
  template DCOO<IT, VT>* dmmio::DCOO_read_f(FILE* f, int comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size, PartitioningType part_type, Operation op, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta, int padding, bool permute, IT * perm_vec); \
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
    bool expl_val_for_bin_mtx, Matrix_Metadata* meta,
    int padding, bool permute, IT * perm_vec
  ) {
    return DCOO_read_f<IT, VT>(
      mmio::io::open_file_r(filename),
      mpi_comm_size, rank,
      grid_rows, grid_cols, grid_node_size,
      partitioning_type, op,
      mmio::io::mm_is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta, padding, permute,
      perm_vec
    );
  }


  template<typename IT, typename VT>
  IT * create_permutation(Entry<IT, VT> * entries, const IT nrows, const IT ncols, const int rank)
  {
    assert(nrows==ncols);
    
    std::random_device rd;
    std::mt19937 g(rd());

    IT * perm = (IT*)malloc(sizeof(IT)*nrows);
    if (rank==0)
    {
        // Generate permutation vector on rank 0
        std::iota(perm, perm + nrows, 0);
        std::shuffle(perm, perm + nrows, g);
    }

    // Give everyone the permutation vector
    MPI_Bcast(perm, nrows, MPI_INT, 0, MPI_COMM_WORLD);

    return perm;
  }


  template<typename IT, typename VT>
  void apply_symmetric_permutation(Entry<IT, VT> * entries, const IT n, IT * perm)
  {
      for (IT i=0; i<n; i++)
      {
          IT rid = entries[i].row;
          IT cid = entries[i].col;
          entries[i].row = perm[rid];
          entries[i].col = perm[cid];
      }
  }


  template<typename IT, typename VT>
  DCOO<IT, VT>* DCOO_read_f(
    FILE *f,
    int mpi_comm_size, int rank,
    int grid_rows, int grid_cols, int grid_node_size,
    PartitioningType partitioning_type, Operation op,
    bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta,
    int padding, bool permute,
    IT * perm_vec) {
    IT nrows, ncols, local_nnz;
    MM_typecode matcode;
    Entry<IT, VT> *entries = dmmio::io::mm_parse_file_distributed<IT, VT>(f, rank, mpi_comm_size, nrows, ncols, local_nnz, &matcode, is_bmtx, meta);

    DCOO<IT, VT> *dcoo = (DCOO<IT, VT>*)malloc(sizeof(DCOO<IT, VT>));
    if (permute) {

        if (perm_vec == nullptr) {
            dcoo->permutation = create_permutation(entries, nrows, ncols, rank);
        } else {
            dcoo->permutation = (IT*)malloc(sizeof(IT) * nrows);
            memcpy(dcoo->permutation, perm_vec, sizeof(IT) * nrows);
        }

        apply_symmetric_permutation(entries, local_nnz, dcoo->permutation);
    }

    // For UINT32_T datatype in the Alltoallv
    static_assert( (sizeof(Entry<IT, VT>) % sizeof(uint32_t) == 0 ));

    // Do padding
    while (nrows % (grid_rows * grid_node_size * padding) != 0 && 
            nrows % (grid_cols * padding)) {
        nrows++;
    }

    while (ncols % (grid_rows * grid_node_size * padding) != 0 && 
            ncols % (grid_cols * padding)) {
        ncols++;
    }

    if (entries == NULL) return NULL;
    Partitioning *partitioning = Partitioning_create(nrows, ncols, grid_rows, grid_cols, grid_node_size, partitioning_type, op);

    dcoo->partitioning = partitioning;
    dcoo->permuted = permute;


    int *owner = (int*)malloc(sizeof(int)*local_nnz);
    for (int i=0; i<local_nnz; i++) owner[i] = dmmio::partitioning::edgeowner::edge2owner(partitioning, entries[i].row, entries[i].col);

    for (int i=0; i<local_nnz; i++) {
        if (owner[i] >= mpi_comm_size || owner[i] < 0) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            printf("Rank %d -- owner[%d]: %d, row: %d, col: %d, val: %f\n", rank, i, owner[i], entries[i].row, entries[i].col, entries[i].val);
            exit(EXIT_FAILURE);
        }
    }


    #ifdef DEBUG
    MPI_ALL_PRINT(
      for (int i=0; i<local_nnz; i++) {
        fprintf(fp, "\t%lu, %lu, %lu --> %d\n", entries[i].row, entries[i].col, entries[i].val, owner[i]);
      }
      fprintf(fp, "\n");
    )
    #endif


    // Sort the entries according to the owner process and rebuild the new owner vector
    Entry<IT, VT>* sorted_entries = sortEntriesByOwner<IT, VT>(entries, owner, local_nnz);
    free(entries);
    free(owner);

    owner = (int*)malloc(sizeof(int)*local_nnz);
    for (int i=0; i<local_nnz; i++) owner[i] = dmmio::partitioning::edgeowner::edge2owner(partitioning, sorted_entries[i].row, sorted_entries[i].col);
    

    #ifdef DEBUG
    MPI_ALL_PRINT(
      fprintf(fp, "Sorted entries:\n");
      for (int i=0; i<local_nnz; i++) {
          fprintf(fp, "\t%lu, %lu, %lu --> %d\n", sorted_entries[i].row, sorted_entries[i].col, sorted_entries[i].val, owner[i]);
      }
      fprintf(fp, "\n");
    )
    #endif

    int* counts_send        = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* counts_recv        = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* displacements_send = (int*)malloc(sizeof(int)*mpi_comm_size);
    int* displacements_recv = (int*)malloc(sizeof(int)*mpi_comm_size);

    for (int i=0; i<local_nnz; i++) {
        if (owner[i] >= mpi_comm_size || owner[i] < 0) {
            printf("owner[%d]: %d, row: %d, col: %d\n", i, owner[i], sorted_entries[i].row, sorted_entries[i].col);
            exit(EXIT_FAILURE);
        }
    }

    for (int i=0; i<mpi_comm_size; i++) counts_send[i] = 0;
    for (int i=0; i<local_nnz; i++) counts_send[owner[i]] += sizeof(Entry<IT, VT>)/sizeof(uint32_t);

    displacements_send[0] = 0;
    for (int i=1; i<mpi_comm_size; i++) displacements_send[i] = displacements_send[i-1] + counts_send[i-1];

    MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

    free(owner);

    displacements_recv[0] = 0;
    for (int i = 1; i < mpi_comm_size; i++) displacements_recv[i] = displacements_recv[i-1] + counts_recv[i-1];
    int total_recv = (displacements_recv[mpi_comm_size-1] + counts_recv[mpi_comm_size-1]) / (sizeof(Entry<IT, VT>)/sizeof(uint32_t));
    Entry<IT, VT>* recv_entries = (Entry<IT, VT>*)malloc(total_recv * sizeof(Entry<IT, VT>));

  // #define DEBUG_ALLTOALLV
  #ifdef DEBUG_ALLTOALLV
    MPI_ALL_PRINT(
        fprintf(fp, "Rank %d of %d\n", rank, mpi_comm_size);
        fprintf(fp, "local_nnz: %d | total_recv: %d\n\n", local_nnz, total_recv);

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
        for (int i = 0; i < local_nnz; i++) {
            fprintf(fp, "\t\t[%3d] -> row: %lu, col: %lu, val: %lu\n",
                    owner[i], sorted_entries[i].row, sorted_entries[i].col, sorted_entries[i].val);
        }
        fprintf(fp, "\n")
    )
  #endif

    MPI_Alltoallv(sorted_entries,
                  counts_send,
                  displacements_send,
                  MPI_UINT32_T,
                  recv_entries,
                  counts_recv,
                  displacements_recv,
                  MPI_UINT32_T,
                  MPI_COMM_WORLD);


    free(counts_send);
    free(counts_recv);
    free(displacements_send);
    free(displacements_recv);

    // TODO keep track of global and local matrix dimensions PROPERLY
    COO<IT, VT> *coo = mmio::COO_create<IT, VT>(nrows, ncols, total_recv, expl_val_for_bin_mtx || !meta->is_pattern);
    mmio::io::Entries_to_COO<IT, VT>(recv_entries, coo);
    dcoo->coo = coo;

    return dcoo;
  }

  template<typename IT, typename VT>
  void DCOO_destroy(DCOO<IT, VT>** dcoo) {
    if ((*dcoo)->permuted) {
        free((*dcoo)->permutation);
    }
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

  template<typename IT, typename VT>
  DDENSE<IT, VT>* dcoo2ddense(DCOO<IT, VT>* dcoo) {
    DDENSE<IT,VT> *dense;
    dense->partitioning = dcoo->partitioning;
    dense->mat = coo2dense(dcoo->coo);
    return(dense);
  };

} // namespace dmmio

DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int64_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int64_t, double)
