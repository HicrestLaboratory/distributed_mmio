#include "../../include/dmmio/dmmio.h"

#include <ccutils/colors.h>
#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <ccutils/macros.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "../../include/dmmio/dio.h"
#include "../../include/dmmio/partitioning.h"
#include "../../include/mmio/io.h"
#include "../../include/mmio/mmio.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
using Operation = dmmio::Operation;
using PartitioningType = dmmio::PartitioningType;
using Partitioning = dmmio::Partitioning;
using ProcessGrid = dmmio::ProcessGrid;

template <typename IT, typename VT>
using Entry = mmio::io::Entry<IT, VT>;
template <typename IT, typename VT>
using DCOO = dmmio::DCOO<IT, VT>;
template <typename IT, typename VT>
using DCSR = dmmio::DCSR<IT, VT>;
template <typename IT, typename VT>
using COO = mmio::COO<IT, VT>;

#define DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(IT, VT)                                                             \
    template DCOO<IT, VT>* dmmio::DCOO_read(                                                                      \
        const char* filename, int mpi_comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size,      \
        PartitioningType partitioning_type, Operation op, bool expl_val_for_bin_mtx, Matrix_Metadata* meta,       \
        bool sort, bool remove_duplicates, bool make_symmetric, bool remove_diagonal, int padding, bool permute, IT* perm_vec);                      \
    template DCOO<IT, VT>* dmmio::DCOO_read_f(                                                                    \
        FILE* f, int comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size,                       \
        PartitioningType part_type, Operation op, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta, \
         bool sort, bool remove_duplicates, bool make_symmetric, bool remove_diagonal, int padding, bool permute, IT* perm_vec);                      \
    template void dmmio::DCOO_destroy(DCOO<IT, VT>** dcoo);                                                       \
    template void dmmio::DCSR_destroy(DCSR<IT, VT>** dcsr);                                                       \
    template DCSR<IT, VT>* dmmio::DCOO2DCSR(DCOO<IT, VT>* dcoo);

namespace dmmio {

template <typename IT, typename VT>
Entry<IT, VT>* sortEntriesByOwner(const Entry<IT, VT>* entries, const int* owner, size_t nentries) {
    // Combine entries and owner into a vector of pairs
    std::vector<std::pair<int, Entry<IT, VT>>> combined(nentries);
    for (size_t i = 0; i < nentries; ++i) {
        combined[i] = {owner[i], entries[i]};
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

Partitioning* Partitioning_create(int matrix_rows, int matrix_cols, int grid_rows, int grid_cols, int grid_node_size,
                                  PartitioningType partitioning_type, Operation operation) {
    Partitioning* partitioning = (Partitioning*)malloc(sizeof(Partitioning));
    ProcessGrid* grid = dmmio::io::ProcessGrid_create(grid_rows, grid_cols, grid_node_size);

    dmmio::io::set_partitioning_type(partitioning, partitioning_type, operation);
    dmmio::io::set_partitioning_grid(partitioning, grid);
    dmmio::io::set_partitioning_global_dim(partitioning, matrix_rows, matrix_cols);
    dmmio::io::set_partitioning_group_dim(partitioning);
    dmmio::io::set_partitioning_local_dim(partitioning);
    dmmio::io::set_partitioning_functions(partitioning);

    return partitioning;
}

void Partitioning_destroy(Partitioning** partitioning) {
    if (partitioning != NULL && *partitioning != NULL) {
        if ((*partitioning)->grid != NULL) {
            MPI_Comm_free(&((*partitioning)->grid->row_comm));
            MPI_Comm_free(&((*partitioning)->grid->col_comm));
            MPI_Comm_free(&((*partitioning)->grid->node_comm));
            free((*partitioning)->grid);
        }
    }
    if (partitioning != NULL)
        free(partitioning);
}

template <typename IT, typename VT>
DCOO<IT, VT>* DCOO_read(const char* filename, int mpi_comm_size, int rank, int grid_rows, int grid_cols,
                        int grid_node_size, PartitioningType partitioning_type, Operation op, bool expl_val_for_bin_mtx,
                        Matrix_Metadata* meta, bool sort, bool remove_duplicates, bool make_symmetric, bool remove_diagonal, int padding, bool permute,
                        IT* perm_vec) {
    return DCOO_read_f<IT, VT>(mmio::io::open_file_r(filename), mpi_comm_size, rank, grid_rows, grid_cols,
                               grid_node_size, partitioning_type, op,
                               mmio::io::mm_is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta,
                                sort,  remove_duplicates, make_symmetric, remove_diagonal, padding, permute, perm_vec);
}

template <typename IT, typename VT>
IT* create_permutation(Entry<IT, VT>* entries, const IT nrows, const IT ncols, const int rank) {
    assert(nrows == ncols);

    std::random_device rd;
    std::mt19937 g(rd());

    IT* perm = (IT*)malloc(sizeof(IT) * nrows);
    if (rank == 0) {
        // Generate permutation vector on rank 0
        std::iota(perm, perm + nrows, 0);
        std::shuffle(perm, perm + nrows, g);
    }

    // Give everyone the permutation vector
    MPI_Bcast(perm, nrows, MPI_INT, 0, MPI_COMM_WORLD);

    return perm;
}

template <typename IT, typename VT>
void apply_symmetric_permutation(Entry<IT, VT>* entries, const IT n, IT* perm) {
    for (IT i = 0; i < n; i++) {
        IT rid = entries[i].row;
        IT cid = entries[i].col;
        entries[i].row = perm[rid];
        entries[i].col = perm[cid];
    }
}

template <typename IT, typename VT>
DCOO<IT, VT>* DCOO_read_f(FILE* f, int mpi_comm_size, int rank, int grid_rows, int grid_cols, int grid_node_size,
                          PartitioningType partitioning_type, Operation op, bool is_bmtx, bool expl_val_for_bin_mtx,
                          Matrix_Metadata* meta, bool sort, bool remove_duplicates,  bool make_symmetric, bool remove_diagonal, int padding, bool permute,
                          IT* perm_vec) {
    IT nrows, ncols, local_nnz;
    MM_typecode matcode;
    Entry<IT, VT>* entries = dmmio::io::mm_parse_file_distributed<IT, VT>(
        f, rank, mpi_comm_size, nrows, ncols, local_nnz, &matcode, is_bmtx, meta, remove_diagonal);
    IT global_nrows = nrows;
    IT global_ncols = ncols;

    DCOO<IT, VT>* dcoo = (DCOO<IT, VT>*)malloc(sizeof(DCOO<IT, VT>));
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
    static_assert((sizeof(Entry<IT, VT>) % sizeof(uint32_t) == 0));

    // Do padding
    while (nrows % (grid_rows * grid_node_size * padding) != 0 && nrows % (grid_cols * padding)) {
        nrows++;
    }

    while (ncols % (grid_rows * grid_node_size * padding) != 0 && ncols % (grid_cols * padding)) {
        ncols++;
    }

    if (entries == NULL)
        return NULL;
    Partitioning* partitioning =
        Partitioning_create(nrows, ncols, grid_rows, grid_cols, grid_node_size, partitioning_type, op);

    dcoo->partitioning = partitioning;
    dcoo->permuted = permute;

    int* owner = (int*)malloc(sizeof(int) * local_nnz);
    for (int i = 0; i < local_nnz; i++)
        owner[i] = dmmio::partitioning::edgeowner::edge2owner(partitioning, entries[i].row, entries[i].col);

    for (int i = 0; i < local_nnz; i++) {
        if (owner[i] >= mpi_comm_size || owner[i] < 0) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            printf("Rank %d -- owner[%d]: %d, row: %d, col: %d, val: %f\n", rank, i, owner[i], entries[i].row,
                   entries[i].col, entries[i].val);
            exit(EXIT_FAILURE);
        }
    }

#ifdef DEBUG
    CCUTILS_MPI_INIT
    CCUTILS_MPI_ALL_PRINT(for (int i = 0; i < local_nnz; i++) {
        fprintf(fp, "\t%lu, %lu, %lu --> %d\n", entries[i].row, entries[i].col, entries[i].val, owner[i]);
    } fprintf(fp, "\n");)
#endif

    // Sort the entries according to the owner process and rebuild the new owner vector
    Entry<IT, VT>* sorted_entries = sortEntriesByOwner<IT, VT>(entries, owner, local_nnz);
    free(entries);
    free(owner);

    owner = (int*)malloc(sizeof(int) * local_nnz);
    for (int i = 0; i < local_nnz; i++)
        owner[i] =
            dmmio::partitioning::edgeowner::edge2owner(partitioning, sorted_entries[i].row, sorted_entries[i].col);

#ifdef DEBUG
    CCUTILS_MPI_ALL_PRINT(fprintf(fp, "Sorted entries:\n"); for (int i = 0; i < local_nnz; i++) {
        fprintf(fp, "\t%lu, %lu, %lu --> %d\n", sorted_entries[i].row, sorted_entries[i].col, sorted_entries[i].val,
                owner[i]);
    } fprintf(fp, "\n");)
#endif

    int* counts_send = (int*)malloc(sizeof(int) * mpi_comm_size);
    int* counts_recv = (int*)malloc(sizeof(int) * mpi_comm_size);
    int* displacements_send = (int*)malloc(sizeof(int) * mpi_comm_size);
    int* displacements_recv = (int*)malloc(sizeof(int) * mpi_comm_size);

    for (int i = 0; i < local_nnz; i++) {
        if (owner[i] >= mpi_comm_size || owner[i] < 0) {
            printf("owner[%d]: %d, row: %d, col: %d\n", i, owner[i], sorted_entries[i].row, sorted_entries[i].col);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < mpi_comm_size; i++)
        counts_send[i] = 0;
    for (int i = 0; i < local_nnz; i++)
        counts_send[owner[i]] += sizeof(Entry<IT, VT>) / sizeof(uint32_t);

    displacements_send[0] = 0;
    for (int i = 1; i < mpi_comm_size; i++)
        displacements_send[i] = displacements_send[i - 1] + counts_send[i - 1];

    MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

    free(owner);

    displacements_recv[0] = 0;
    for (int i = 1; i < mpi_comm_size; i++)
        displacements_recv[i] = displacements_recv[i - 1] + counts_recv[i - 1];
    int total_recv = (displacements_recv[mpi_comm_size - 1] + counts_recv[mpi_comm_size - 1]) /
                     (sizeof(Entry<IT, VT>) / sizeof(uint32_t));
    Entry<IT, VT>* recv_entries = (Entry<IT, VT>*)malloc(total_recv * sizeof(Entry<IT, VT>));

// #define DEBUG_ALLTOALLV
#ifdef DEBUG_ALLTOALLV
    CCUTILS_MPI_ALL_PRINT(
        fprintf(fp, "Rank %d of %d\n", rank, mpi_comm_size);
        fprintf(fp, "local_nnz: %d | total_recv: %d\n\n", local_nnz, total_recv);

        // Print counts_send
        fprintf(fp, "\tcounts_send:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", counts_send[i]); fprintf(fp, "\n");

        // Print displacements_send
        fprintf(fp, "\tdisplacements_send:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", displacements_send[i]); fprintf(fp, "\n");

        // Print counts_recv
        fprintf(fp, "\tcounts_recv:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", counts_recv[i]); fprintf(fp, "\n");

        // Print displacements_recv
        fprintf(fp, "\tdisplacements_recv:\n\t\t%10s ", "value");
        for (int i = 0; i < mpi_comm_size; i++) fprintf(fp, "%3d ", displacements_recv[i]); fprintf(fp, "\n\n");

        // Optionally, print the sorted entries being sent
        fprintf(fp, "\tEntries being sent (sorted by owner):\n"); for (int i = 0; i < local_nnz; i++) {
            fprintf(fp, "\t\t[%3d] -> row: %lu, col: %lu, val: %lu\n", owner[i], sorted_entries[i].row,
                    sorted_entries[i].col, sorted_entries[i].val);
        } fprintf(fp, "\n"))
#endif

    MPI_Alltoallv(sorted_entries, counts_send, displacements_send, MPI_UINT32_T, recv_entries, counts_recv,
                  displacements_recv, MPI_UINT32_T, MPI_COMM_WORLD);

    free(counts_send);
    free(counts_recv);
    free(displacements_send);
    free(displacements_recv);


    if (make_symmetric) {
        const int comm_size = mpi_comm_size;
        const int comm_rank = rank;

        /* ============================================================
         * 1. Count total entries including symmetry
         * ============================================================ */
        IT extra = 0;
        for (IT i = 0; i < total_recv; ++i)
            extra += (recv_entries[i].row != recv_entries[i].col);

        const IT expanded_total = total_recv + extra;

        /* ============================================================
         * 2. Create MPI datatype for Entry
         * ============================================================ */
        MPI_Datatype MPI_ENTRY_TYPE;
        MPI_Type_contiguous(sizeof(Entry<IT, VT>), MPI_BYTE, &MPI_ENTRY_TYPE);
        MPI_Type_commit(&MPI_ENTRY_TYPE);

        /* ============================================================
         * 3. First pass: count sends for ALL entries
         * ============================================================ */
        std::vector<int> send_counts(comm_size, 0);

        for (IT i = 0; i < total_recv; ++i) {
            const auto& e = recv_entries[i];

            // original
            int owner = dmmio::partitioning::edgeowner::edge2owner(partitioning, e.row, e.col);

            if (owner != comm_rank) {
                ++send_counts[owner];
            }

            // symmetric
            if (e.row != e.col) {
                owner = dmmio::partitioning::edgeowner::edge2owner(partitioning, e.col, e.row);

                if (owner != comm_rank)
                    ++send_counts[owner];
            }
        }

        /* ============================================================
         * 4. Send displacements
         * ============================================================ */
        std::vector<int> send_displs(comm_size, 0);
        for (int i = 1; i < comm_size; ++i)
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];

        const int total_send = send_displs.back() + send_counts.back();

        Entry<IT, VT>* send_buf = static_cast<Entry<IT, VT>*>(malloc(total_send * sizeof(Entry<IT, VT>)));

        std::vector<int> cursor = send_displs;

        /* ============================================================
         * 5. Pack ALL mis-owned entries
         * ============================================================ */
        for (IT i = 0; i < total_recv; ++i) {
            const auto& e = recv_entries[i];

            // original
            int owner = dmmio::partitioning::edgeowner::edge2owner(partitioning, e.row, e.col);

            if (owner != comm_rank) {
                send_buf[cursor[owner]++] = e;
            }

            // symmetric
            if (e.row != e.col) {
                Entry<IT, VT> sym{e.col, e.row, e.val};

                owner = dmmio::partitioning::edgeowner::edge2owner(partitioning, sym.row, sym.col);

                if (owner != comm_rank)
                    send_buf[cursor[owner]++] = sym;
            }
        }

        /* ============================================================
         * 6. Exchange counts
         * ============================================================ */
        std::vector<int> recv_counts(comm_size, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        /* ============================================================
         * 7. Receive displacements
         * ============================================================ */
        std::vector<int> recv_displs(comm_size, 0);
        for (int i = 1; i < comm_size; ++i)
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];

        const int total_recv_new = recv_displs.back() + recv_counts.back();

        Entry<IT, VT>* recv_new = static_cast<Entry<IT, VT>*>(malloc(total_recv_new * sizeof(Entry<IT, VT>)));

        /* ============================================================
         * 8. Exchange ALL mis-owned entries
         * ============================================================ */
        MPI_Alltoallv(send_buf, send_counts.data(), send_displs.data(), MPI_ENTRY_TYPE, recv_new, recv_counts.data(),
                      recv_displs.data(), MPI_ENTRY_TYPE, MPI_COMM_WORLD);

        free(send_buf);

        /* ============================================================
         * 9. Count locally owned entries
         * ============================================================ */
        IT local_count = 0;
        for (IT i = 0; i < total_recv; ++i) {
            const auto& e = recv_entries[i];
            if (dmmio::partitioning::edgeowner::edge2owner(partitioning, e.row, e.col) == comm_rank)
                ++local_count;

            if (e.row != e.col && dmmio::partitioning::edgeowner::edge2owner(partitioning, e.col, e.row) == comm_rank)
                ++local_count;
        }

        /* ============================================================
         * 10. Final buffer
         * ============================================================ */
        Entry<IT, VT>* final_entries =
            static_cast<Entry<IT, VT>*>(malloc((local_count + total_recv_new) * sizeof(Entry<IT, VT>)));

        IT pos = 0;

        // locally-owned originals + symmetric
        for (IT i = 0; i < total_recv; ++i) {
            const auto& e = recv_entries[i];

            if (dmmio::partitioning::edgeowner::edge2owner(partitioning, e.row, e.col) == comm_rank)
                final_entries[pos++] = e;

            if (e.row != e.col && dmmio::partitioning::edgeowner::edge2owner(partitioning, e.col, e.row) == comm_rank)
                final_entries[pos++] = Entry<IT, VT>{e.col, e.row, e.val};
        }

        // received entries
        memcpy(final_entries + pos, recv_new, total_recv_new * sizeof(Entry<IT, VT>));
        pos += total_recv_new;

        free(recv_entries);
        free(recv_new);

        recv_entries = final_entries;
        total_recv = pos;
        local_nnz = pos;

        /* ============================================================
         * 11. Cleanup MPI datatype
         * ============================================================ */
        MPI_Type_free(&MPI_ENTRY_TYPE);
    }

    COO<IT, VT>* coo = mmio::COO_create<IT, VT>(nrows, ncols, total_recv, expl_val_for_bin_mtx || !meta->is_pattern);
    mmio::io::Entries_to_COO<IT, VT>(recv_entries, coo);
    mmio::COO_sort_and_deduplicate(coo, sort, remove_duplicates);

    dcoo->nrows = global_nrows;
    dcoo->ncols = global_ncols;
    dcoo->coo = coo;
    IT global_nnz;
    MPI_Allreduce(&(coo->nnz), &global_nnz, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
    dcoo->nnz = global_nnz;

    return dcoo;
}

template <typename IT, typename VT>
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

template <typename IT, typename VT>
void DCSR_destroy(DCSR<IT, VT>** dcsr) {
    if ((*dcsr)->permuted && (*dcsr)->permutation != NULL) {
        free((*dcsr)->permutation);
    }
    if (dcsr != NULL && *dcsr != NULL) {
        mmio::CSR_destroy(&((*dcsr)->csr));
        Partitioning_destroy(&((*dcsr)->partitioning));
    }
}

template <typename IT, typename VT>
DDENSE<IT, VT>* dcoo2ddense(DCOO<IT, VT>* dcoo) {
    DDENSE<IT, VT>* dense;
    dense->partitioning = dcoo->partitioning;
    dense->mat = coo2dense(dcoo->coo);
    return (dense);
};

template <typename IT, typename VT>
DCSR<IT, VT>* DCOO2DCSR(DCOO<IT, VT>* dcoo) {
    CCUTILS_ASSERT(dcoo->partitioning->grid->col_size == 1 || dcoo->partitioning->grid->row_size == 1,
                   "DCOO_2_DCSR currently only supports 1D partitionings\n")
    // TODO ensure partitioning maj dim is compatible with rows
    DCSR<IT, VT>* dcsr = (DCSR<IT, VT>*)malloc(sizeof(DCSR<IT, VT>));
    dcsr->nrows = dcoo->nrows;
    dcsr->ncols = dcoo->ncols;
    dcsr->nnz = dcoo->nnz;
    // TODO deep copy partitioning
    dcsr->partitioning = dcoo->partitioning;
    // TODO handle permutations

    dcsr->csr = mmio::COO2CSR(dcoo->coo, false);

    return dcsr;
}

}  // namespace dmmio

DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int64_t, float)
DMMIO_DSTRUCTS_EXPLICIT_TEMPLATE_INST(int64_t, double)
