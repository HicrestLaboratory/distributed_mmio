#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <climits>
#include <vector>

#include "../../include/dmmio/dmmio.h"
#include "../../include/mmio/utils.h"

template <typename IT, typename VT>
using DCOO = dmmio::DCOO<IT, VT>;

#define DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(IT, VT) \
    template void dmmio::utils::DCOO_print_as_dense(DCOO<IT, VT>* dcoo, std::string header, FILE* fp);

namespace dmmio::utils {

void ProcessGrid_print(const dmmio::ProcessGrid* grid, FILE* fp) {
    if (grid->global_rank == 0) {
        fprintf(fp, "========================\n");
        fprintf(fp, " ProcessGrid Details \n");
        fprintf(fp, "========================\n");
        fprintf(fp, "Total processes:\t %d\n", grid->global_size);
        fprintf(fp, "row size:\t %d\n", grid->row_size);
        fprintf(fp, "col size:\t %d\n", grid->col_size);
        fprintf(fp, "node size:\t %d\n", grid->node_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);

    for (int i = 0; i < grid->global_size; i++) {
        if (grid->global_rank == i) {
            fprintf(fp, "----- Process %d -----\n", grid->global_rank);
            fprintf(fp, "Rank:\t %d\n", grid->global_rank);
            fprintf(fp, "row rank:\t %d\n", grid->row_rank);
            fprintf(fp, "col rank:\t %d\n", grid->col_rank);
            fprintf(fp, "node rank:\t %d\n", grid->node_rank);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    sleep(1);
    if (grid->global_rank == 0)
        fprintf(fp, "========================\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

typedef struct {
    int col_rank;
    int row_rank;
    int node_rank;

    int hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    int gid;
} nodeInfo;

nodeInfo* genNodeInfo(const dmmio::ProcessGrid* grid) {
    nodeInfo* myinfo = (nodeInfo*)malloc(sizeof(nodeInfo));
    myinfo->col_rank = grid->col_rank;
    myinfo->row_rank = grid->row_rank;
    myinfo->node_rank = grid->node_rank;

    MPI_Get_processor_name(myinfo->hostname, &(myinfo->hostname_len));
    return (myinfo);
}

void genEmptyNodeInfo(nodeInfo* emptyinfo) {
    emptyinfo->col_rank = -1;
    emptyinfo->row_rank = -1;
    emptyinfo->node_rank = -1;

    const char* msg = "Uninitialized";
    emptyinfo->hostname_len = strlen(msg);
    snprintf(emptyinfo->hostname, MPI_MAX_PROCESSOR_NAME, "%s", msg);
}

void overwriteNodeInfo(nodeInfo* destination, nodeInfo* source) {
    destination->col_rank = source->col_rank;
    destination->row_rank = source->row_rank;
    destination->node_rank = source->node_rank;

    destination->hostname_len = source->hostname_len;
    memcpy(destination->hostname, source->hostname, sizeof(char) * MPI_MAX_PROCESSOR_NAME);
}

int ProcessGrid_graph(const dmmio::ProcessGrid* grid, FILE* fp, bool host_gpu_id_print) {
    int row_size = grid->row_size;
    int col_size = grid->col_size;
    int node_size = grid->node_size;
    int global_size = grid->global_size;

    // Allocate 3D grid dynamically: [row][col][node]
    nodeInfo*** tmp_grid = (nodeInfo***)malloc(col_size * sizeof(nodeInfo**));
    for (int row = 0; row < col_size; ++row) {
        tmp_grid[row] = (nodeInfo**)malloc(row_size * sizeof(nodeInfo*));
        for (int col = 0; col < row_size; ++col) {
            tmp_grid[row][col] = (nodeInfo*)malloc(node_size * sizeof(nodeInfo));
            // Initialize with -1 (or any invalid rank)
            for (int node = 0; node < node_size; ++node) {
                genEmptyNodeInfo(&(tmp_grid[row][col][node]));
            }
        }
    }

    // Populate the grid
    /*
    int myvalues[3];
    myvalues[0] = grid->col_rank;
    myvalues[1] = grid->row_rank;
    myvalues[2] = grid->node_rank;
    int *allvalues = (int*)malloc(sizeof(int)*global_size*3);
    */
    nodeInfo* myinfo = genNodeInfo(grid);
    nodeInfo* allinfo = (nodeInfo*)malloc(sizeof(nodeInfo) * global_size);
    MPI_Allgather(myinfo, sizeof(nodeInfo), MPI_BYTE, allinfo, sizeof(nodeInfo), MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < global_size; i++) {
        // tmp_grid[allvalues[3*i]][allvalues[3*i+1]][allvalues[3*i+2]] = i;
        overwriteNodeInfo(&(tmp_grid[allinfo[i].col_rank][allinfo[i].row_rank][allinfo[i].node_rank]), &(allinfo[i]));
        tmp_grid[allinfo[i].col_rank][allinfo[i].row_rank][allinfo[i].node_rank].gid = i;
    }

    int max_hostnamelen = 0;
    MPI_Allreduce(&(myinfo->hostname_len), &max_hostnamelen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Header row: column labels
    if (grid->global_rank == 0)
        fprintf(fp, "         ");
    for (int col = 0; col < row_size; ++col) {
        if (grid->global_rank == 0)
            fprintf(fp, "col %-2d         ", col);
    }
    if (grid->global_rank == 0)
        fprintf(fp, "\n");

    // Top border
    if (grid->global_rank == 0)
        fprintf(fp, "       ");
    for (int col = 0; col < row_size; ++col) {
        if (grid->global_rank == 0)
            fprintf(fp, "------------------- ");
    }
    if (grid->global_rank == 0)
        fprintf(fp, "\n");

    // For each row
    for (int row = 0; row < col_size; ++row) {
        for (int node = 0; node < node_size; ++node) {
            if (node == 0) {
                if (grid->global_rank == 0)
                    fprintf(fp, "row %-2d |", row);
            } else {
                if (grid->global_rank == 0)
                    fprintf(fp, "       |");
            }

            for (int col = 0; col < row_size; ++col) {
                int gid = tmp_grid[row][col][node].gid;
                char* hostname = tmp_grid[row][col][node].hostname;
                if (grid->global_rank == 0) {
                    if (host_gpu_id_print) {
                        fprintf(fp, " Node%d [%-3d] [%12s]   |", node, gid, hostname);
                    } else {
                        fprintf(fp, " Node%d [%-3d]     |", node, gid);
                    }
                }
            }
            if (grid->global_rank == 0)
                fprintf(fp, "\n");
        }

        // Separator line
        if (grid->global_rank == 0)
            fprintf(fp, "       ");
        for (int col = 0; col < row_size; ++col) {
            if (grid->global_rank == 0)
                fprintf(fp, "------------------- ");
        }
        if (grid->global_rank == 0)
            fprintf(fp, "\n");
    }

    // Cleanup memory
    for (int row = 0; row < col_size; ++row) {
        for (int col = 0; col < row_size; ++col) {
            free(tmp_grid[row][col]);
        }
        free(tmp_grid[row]);
    }
    free(tmp_grid);
    free(myinfo);

    return 0;
}

template <typename IT, typename VT>
mmio::COO<IT, VT>* DCOO_gather(dmmio::DCOO<IT, VT>* dcoo, int gather_on_rank = 0) {
    if (!dcoo || !dcoo->partitioning) {
        return nullptr;
    }

    MPI_Comm comm = dcoo->partitioning->grid->world_comm;
    int comm_rank = dcoo->partitioning->grid->global_rank;
    int comm_size = dcoo->partitioning->grid->global_size;

    /* ============================================================
     * 1. Determine if this is a pattern matrix
     * ============================================================ */
    bool is_pattern = (dcoo->coo && dcoo->coo->val == nullptr);
    
    /* ============================================================
     * 2. Create MPI datatypes
     * ============================================================ */
    MPI_Datatype MPI_IT, MPI_VT;
    bool free_IT = false, free_VT = false;
    
    if (sizeof(IT) == 4) {
        MPI_IT = MPI_INT32_T;
    } else if (sizeof(IT) == 8) {
        MPI_IT = MPI_INT64_T;
    } else {
        MPI_Type_contiguous(sizeof(IT), MPI_BYTE, &MPI_IT);
        MPI_Type_commit(&MPI_IT);
        free_IT = true;
    }
    
    if (!is_pattern) {
        if (sizeof(VT) == 4) {
            MPI_VT = MPI_FLOAT;
        } else if (sizeof(VT) == 8) {
            MPI_VT = MPI_DOUBLE;
        } else {
            MPI_Type_contiguous(sizeof(VT), MPI_BYTE, &MPI_VT);
            MPI_Type_commit(&MPI_VT);
            free_VT = true;
        }
    }

    /* ============================================================
     * 3. Gather local nnz counts
     * ============================================================ */
    IT local_nnz = dcoo->coo ? dcoo->coo->nnz : 0;
    
    // Check for overflow when converting to int
    if (local_nnz > INT_MAX) {
        fprintf(stderr, "Error: local_nnz exceeds INT_MAX\n");
        MPI_Abort(comm, 1);
    }
    
    int local_nnz_int = static_cast<int>(local_nnz);
    
    // All ranks need these arrays for MPI_Gatherv
    std::vector<int> nnz_counts(comm_size);
    std::vector<int> nnz_displs(comm_size);
    
    MPI_Gather(&local_nnz_int, 1, MPI_INT, 
               nnz_counts.data(), 1, MPI_INT, 
               gather_on_rank, comm);

    /* ============================================================
     * 4. Calculate displacements and total nnz (all ranks)
     * ============================================================ */
    nnz_displs[0] = 0;
    for (int i = 1; i < comm_size; ++i) {
        nnz_displs[i] = nnz_displs[i - 1] + nnz_counts[i - 1];
    }
    IT total_nnz = nnz_displs[comm_size - 1] + nnz_counts[comm_size - 1];

    /* ============================================================
     * 5. Allocate gathered COO on target rank
     * ============================================================ */
    mmio::COO<IT, VT>* gathered_coo = nullptr;
    IT* gathered_rows = nullptr;
    IT* gathered_cols = nullptr;
    VT* gathered_vals = nullptr;
    
    if (comm_rank == gather_on_rank) {
        gathered_coo = new mmio::COO<IT, VT>();
        gathered_coo->nrows = dcoo->nrows;
        gathered_coo->ncols = dcoo->ncols;
        gathered_coo->nnz = total_nnz;
        
        if (total_nnz > 0) {
            gathered_coo->row = static_cast<IT*>(malloc(total_nnz * sizeof(IT)));
            gathered_coo->col = static_cast<IT*>(malloc(total_nnz * sizeof(IT)));
            
            gathered_rows = gathered_coo->row;
            gathered_cols = gathered_coo->col;
            
            // Only allocate values for non-pattern matrices
            if (!is_pattern) {
                gathered_coo->val = static_cast<VT*>(malloc(total_nnz * sizeof(VT)));
                gathered_vals = gathered_coo->val;
            } else {
                gathered_coo->val = nullptr;
            }
        } else {
            gathered_coo->row = nullptr;
            gathered_coo->col = nullptr;
            gathered_coo->val = nullptr;
        }
    }

    /* ============================================================
     * 6. Prepare send buffers
     * ============================================================ */
    IT* local_rows = (dcoo->coo && local_nnz > 0) ? dcoo->coo->row : nullptr;
    IT* local_cols = (dcoo->coo && local_nnz > 0) ? dcoo->coo->col : nullptr;
    VT* local_vals = (dcoo->coo && local_nnz > 0 && !is_pattern) ? dcoo->coo->val : nullptr;

    /* ============================================================
     * 7. Gather row indices
     * ============================================================ */
    MPI_Gatherv(local_rows, local_nnz_int, MPI_IT,
                gathered_rows, nnz_counts.data(), nnz_displs.data(), 
                MPI_IT, gather_on_rank, comm);

    /* ============================================================
     * 8. Gather column indices
     * ============================================================ */
    MPI_Gatherv(local_cols, local_nnz_int, MPI_IT,
                gathered_cols, nnz_counts.data(), nnz_displs.data(),
                MPI_IT, gather_on_rank, comm);

    /* ============================================================
     * 9. Gather values (only for non-pattern matrices)
     * ============================================================ */
    if (!is_pattern) {
        MPI_Gatherv(local_vals, local_nnz_int, MPI_VT,
                    gathered_vals, nnz_counts.data(), nnz_displs.data(),
                    MPI_VT, gather_on_rank, comm);
    }

    /* ============================================================
     * 10. Cleanup custom MPI datatypes if created
     * ============================================================ */
    if (free_IT) {
        MPI_Type_free(&MPI_IT);
    }
    if (free_VT) {
        MPI_Type_free(&MPI_VT);
    }

    return gathered_coo;
}

template <typename IT, typename VT>
void DCOO_print_as_dense(dmmio::DCOO<IT, VT>* dcoo, std::string header, FILE* fp) {
    const int gather_on_rank = 0;
    mmio::COO<IT, VT>* coo = DCOO_gather(dcoo, gather_on_rank);
    if (coo) {
        mmio::utils::COO_print_as_dense(coo, header, fp);
        mmio::COO_destroy(&coo);
    }
}

}  // namespace dmmio::utils

DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(int64_t, float)
DMMIO_UTILS_EXPLICIT_TEMPLATE_INST(int64_t, double)