#include "dist_csr.cuh"
#include "combblas_mmio.h"


DistCSR * make_dist_csr(myLocalCSR * my_local_csr, Partitioning *part,
                        uint64_t global_nrows, uint64_t global_ncols, uint64_t global_nnz)
{

    DistCSR * distCSR = (DistCSR*)(Malloc(sizeof(DistCSR)));
    ProcessGrid *grid = part->proc_grid;

    distCSR->global_nrows = global_nrows;
    distCSR->global_ncols = global_ncols;
    distCSR->global_nnz = global_nnz;

    distCSR->my_local_csr = my_local_csr;
    distCSR->mypart = part;
    distCSR->grid = grid;

    return distCSR;
}


DistCSR * read_dist_csr(const char * fpath, Partitioning *part, PartitionFunction part_f, int transpose, int part_balanced, int testpartitioning, Scrambling * scramble)
{
    // Metadata
    uint64_t * edges, deleted;
    ProcessGrid * grid = part->proc_grid;
    float * values;
    ull nnodes;
    uint64_t global_nnz, global_nrows, global_ncols, ndel;

    // Read in the edgelist
    std::string filename(fpath);
    uint64_t nedges = read_dist_mat(filename,
                                    &edges, &values, 
                                    &global_nrows, &global_ncols,
                                    part, part_f, transpose, scramble);

    if (testpartitioning) {
        if(grid->rank == 0) fprintf(stdout, "WARNING: %s function was called with testpartitioning flag on. This means that the mtx values will be deleted and changed with values required by some correctness tests. If you are not in debug mode, please, call this function without this flag.\n", __func__);
        for (int k=0; k<nedges; k++) {
            values[k] = ENCODE_COO2VAL(edges[2*k], edges[2*k+1]);
        }
    }

    uint64_t total_edges = 0;
    MPI_Allreduce(&nedges, &total_edges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

#if DEBUG
    fprintf(stdout, "Process %d has %llu edges\n", grid->rank, nedges);

    if(grid->rank==0) fprintf(stdout, "\n========================== Setting up local CSR ==========================\n");
    fflush(stdout);
    sleep(0.1);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Compute local matrix dimensions
    uint64_t loc_nrows, loc_ncols;
    compute_local_matrix_dims(part,
                              global_nrows, global_ncols, 
                              &loc_nrows, &loc_ncols);

    // Build local CSR 
    myLocalCSR * local_csr = (myLocalCSR*)(Malloc(sizeof(myLocalCSR)));
    transpose = !part_balanced && transpose;
    build_local_csr_gpu(part, edges, values, nedges, loc_nrows, loc_ncols, (bool)transpose, local_csr);

#if DEBUG
    if(grid->rank==0) fprintf(stdout, "\n========================== Setting up dist CSR ==========================\n");
    fflush(stdout);
    sleep(0.1);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Setup DistCSR
    DistCSR * distCSR = make_dist_csr(local_csr, part, global_nrows, global_ncols, total_edges);

#if DEBUG
    if(grid->rank==0) fprintf(stdout, "\n========================== Done setting up dist CSR ==========================\n");
    fflush(stdout);
    sleep(0.1);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return distCSR;
}



bool equals(DistCSR * lhs, DistCSR * rhs)
{
    int _equals = (int)equals(lhs->my_local_csr, rhs->my_local_csr);
    if (_equals==0)
        fprintf(stderr, "Incorrect on rank %d\n", lhs->grid->rank);
    MPI_Allreduce(MPI_IN_PLACE, &_equals, 1, MPI_INT, MPI_LAND, lhs->grid->world_comm);
    return _equals;
}


DistCSR * copy(DistCSR * other, bool set_nzr)
{
    DistCSR * result = (DistCSR*)Malloc(sizeof(DistCSR));

    result->my_local_csr = copy(other->my_local_csr, set_nzr);
    result->global_nrows = other->global_nrows;

    result->global_nrows = other->global_nrows;
    result->global_ncols = other->global_ncols;
    result->global_nnz = other->global_nnz;

    result->mypart = (Partitioning * )malloc(sizeof(Partitioning));
    set_mypart_type(result->mypart, other->mypart->my_part_type, other->mypart->operand_type);
    set_mypart_grid(result->mypart, other->grid);
    result->grid = other->grid; // DistCSR does not own process grid
    return result;
}


DistCSR * move(DistCSR * other)
{
    DistCSR * result = (DistCSR*)(Malloc(sizeof(DistCSR)));
    result->my_local_csr = move(other->my_local_csr);
    result->global_nrows = other->global_nrows;

    result->global_nrows = other->global_nrows;
    result->global_ncols = other->global_ncols;
    result->global_nnz = other->global_nnz;
    result->grid = other->grid; // DistCSR does not own process grid
    result->mypart = other->mypart;
    other->mypart = NULL;
    free(other);

    return result;
}


void prune(DistCSR * A, float tol, cusparseHandle_t * handle)
{
    A->my_local_csr = prune(A->my_local_csr, tol, handle);
    MPI_Allreduce(&A->my_local_csr->nnz, &A->global_nnz, 1, MPI_UINT64_T, MPI_SUM, A->grid->world_comm);
}


void prune_slow(DistCSR * A, float tol, cusparseHandle_t * handle)
{
    A->my_local_csr = prune_slow(A->my_local_csr, tol, handle);
    MPI_Allreduce(&A->my_local_csr->nnz, &A->global_nnz, 1, MPI_UINT64_T, MPI_SUM, A->grid->world_comm);
}


void write_stats(DistCSR * distCSR, const char * fname)
{
    myLocalCSR * M = distCSR->my_local_csr;
    ProcessGrid * grid = distCSR->grid;
    int rank = grid->rank;
    int n_stats = 3;

    uint64_t send_buf[n_stats] = {M->nnz, M->nnz_r, M->nrows};
    uint64_t * recv_buf = (rank==0) ? (uint64_t*)(malloc(sizeof(uint64_t) * n_stats * grid->nprocs)) : NULL;
    MPI_Gather(send_buf, n_stats, MPI_UINT64_T, recv_buf, n_stats, MPI_UINT64_T, 0, grid->world_comm);

    if (rank==0)
    {

        /* Write to csv */
        FILE * f = fopen(fname, "w");
        fprintf(f, "Rank,nnz,nnzr,nrows\n");
        
        for (int i=0; i<grid->nprocs; i++)
        {
            fprintf(f, "%d,");
            for (int j=0; j<n_stats; j++)
            {
                fprintf(f, "%lu,", recv_buf[j + n_stats * i]);
            }
            fprintf(f, "\n");
        }

        fclose(f);
        free(recv_buf);
    }
}



void print_dist_csr_float(DistCSR * distCSR)
{
    int ntask = distCSR->grid->nprocs;

    for (int i=0; i<ntask; i++) {
        if (distCSR->grid->rank == i) {
            fprintf(stdout, "----- Process %d -----\n", distCSR->grid->rank);
            if (distCSR->global_nnz < 260) {
                print_local_csr<float>(distCSR->my_local_csr);
			}
			fprintf(stdout, "Local nrows:%" PRIu64 "Local ncols:%" PRIu64 "Local nnz:%" PRIu64 " \n",
                    distCSR->my_local_csr->nrows, distCSR->my_local_csr->ncols, distCSR->my_local_csr->nnz);
            MPI_Barrier(MPI_COMM_WORLD);
		}
	}

    MPI_Barrier(MPI_COMM_WORLD);
}


void free_dist_csr(DistCSR * distCSR)
{
    if (distCSR != NULL)
    {
        free_local_csr(distCSR->my_local_csr);
        free(distCSR);
    }
}


